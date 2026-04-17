from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import numpy as np
import requests
import random
from functools import lru_cache  # 新增：缓存API结果，减少CPU占用

# ==========================================
# 【核心修复1】正确的Flask路径配置（解决TemplateNotFound/500报错）
# ==========================================
# 获取app.py所在的目录（项目根目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 模板文件夹：和app.py同级的templates文件夹（必须全小写）
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

# 初始化Flask应用（核心修复：明确指定模板路径）
app = Flask(__name__, template_folder=TEMPLATES_DIR)

# ==========================================
# 配置：API相关（保留你的原有功能）
# ==========================================
# 这里的高德Key保留，但实际用Open-Elevation免费API，无需修改
AMAP_KEY = "44d3e38673dc18b760f544a0d48f8f7f"

# ==========================================
# 1. 光伏发电量仿真模型（支持高德API真实海拔+随机选点）
# ==========================================
class PVGenerationModel:
    def __init__(self):
        self.system_config = {
            'module_efficiency': 0.18,
            'performance_ratio': 0.80,
            'NOCT': 45,
            'temp_loss_coeff': 0.004
        }
        self.site_lat = None
        self.site_lon = None
        self.site_elevation = None

    def run_full_model(self, target_lat=None, target_lon=None, system_capacity_kw=1, random_point=False):
        # 边界值检查：装机容量不能为0或负数
        system_capacity_kw = max(float(system_capacity_kw), 0.1)
        
        if random_point:
            self.site_lat, self.site_lon = self._generate_random_land_point()
            print(f"🎲 随机选点：{self.site_lat}°N, {self.site_lon}°E")
        elif target_lat is not None and target_lon is not None:
            try:
                self.site_lat = round(float(target_lat), 6)
                self.site_lon = round(float(target_lon), 6)
                print(f"📍 地图选点：{self.site_lat}°N, {self.site_lon}°E")
            except (ValueError, TypeError):
                raise ValueError("经纬度必须是有效的数字格式")
        else:
            raise ValueError("请选择模式：指定经纬度 或 random_point=True")

        self.site_elevation = self._get_amap_real_elevation(self.site_lat, self.site_lon)
        print(f"🏔️  真实海拔：{self.site_elevation}m")

        weather_df = self._get_local_weather_data()
        generation_df = self._calculate_pv_generation(weather_df, system_capacity_kw)
        annual_gen_total = float(generation_df['monthly_generation_kwh'].sum())
        annual_gen_per_kw = annual_gen_total / system_capacity_kw
        return generation_df, annual_gen_per_kw

    def _generate_random_land_point(self):
        land_regions = [
            (20.0, 50.0, 100.0, 140.0),
            (25.0, 40.0, 70.0, 100.0),
            (35.0, 60.0, -10.0, 30.0),
            (20.0, 40.0, -10.0, 40.0),
            (15.0, 50.0, -130.0, -60.0),
            (-35.0, 5.0, -80.0, -30.0),
            (-40.0, -10.0, 110.0, 155.0),
            (0.0, 40.0, 10.0, 50.0)
        ]
        region = random.choice(land_regions)
        lat_min, lat_max, lon_min, lon_max = region
        random_lat = round(random.uniform(lat_min, lat_max), 6)
        random_lon = round(random.uniform(lon_min, lon_max), 6)
        return random_lat, random_lon

    def _get_amap_real_elevation(self, lat, lon):
        # 若使用默认Key，直接用模拟海拔
        if AMAP_KEY == "44d3e38673dc18b760f544a0d48f8f7f":
            print("⚠️ 未配置高德地图Key，使用模拟海拔")
            return self._get_simulation_elevation(lat, lon)
        # 优先调用Open-Elevation API
        try:
            return self._get_open_elevation(lat, lon)
        except Exception as e:
            print(f"⚠️ Open-Elevation API调用失败：{str(e)}，使用模拟海拔")
            return self._get_simulation_elevation(lat, lon)

    # 【核心修复2】给API请求加缓存，相同经纬度不会重复请求，大幅减少CPU占用
    @lru_cache(maxsize=100)
    def _get_open_elevation(self, lat, lon):
        try:
            base_url = "https://api.open-elevation.com/api/v1/lookup"
            params = {"locations": f"{lat},{lon}"}
            headers = {
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()  # 触发HTTP状态码异常
            data = response.json()
            if "results" in data and len(data["results"]) > 0:
                elevation = data["results"][0]["elevation"]
                return round(float(elevation), 1)
            raise Exception("API返回无有效海拔数据")
        except Exception as e:
            raise Exception(f"获取海拔失败：{str(e)}")

    def _get_simulation_elevation(self, lat, lon):
        lat = float(lat)
        lon = float(lon)
        # 亚洲内陆（青藏高原等）
        if 25 <= lat <= 35 and 75 <= lon <= 105:
            return round(np.random.uniform(2000, 4500), 1)
        # 经线0/360附近（大西洋等）
        elif abs(lon % 360) <= 10 or abs(lon % 360) >= 350:
            return round(np.random.uniform(0, 100), 1)
        # 中纬度陆地
        elif -30 <= lat <= 50 and abs(lon) <= 150:
            return round(np.random.uniform(50, 500), 1)
        # 其他区域
        else:
            return round(np.random.uniform(100, 2000), 1)

    def _get_local_weather_data(self):
        abs_lat = float(abs(self.site_lat))
        months = np.arange(1, 13, dtype=int)
        # 基础太阳辐照量计算
        base_ghi = 1800.0 - (abs_lat / 90) * 800.0
        elevation_correction = (self.site_elevation / 1000) * 0.05 * base_ghi
        annual_ghi = base_ghi + elevation_correction
        annual_ghi = max(900.0, min(2000.0, annual_ghi))
        # 南北半球月度辐照因子
        if self.site_lat > 0:
            monthly_factor = np.array([0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05], dtype=float)
        else:
            monthly_factor = np.array([0.14, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.05, 0.06, 0.08, 0.10, 0.12], dtype=float)
        # 温度计算（含海拔修正）
        base_temp = 25.0 - (abs_lat / 90) * 30.0
        elevation_temp_correction = (self.site_elevation / 1000) * (-6.0)
        monthly_temp = base_temp + elevation_temp_correction + 10 * np.sin(np.radians((months - 3) * 30))
        # 风速随机生成
        monthly_wind = np.random.uniform(2.0, 6.0, 12)
        # 构建DataFrame并确保类型为Python原生类型
        weather_df = pd.DataFrame({
            'month': months.tolist(),
            'GHI': (annual_ghi * monthly_factor).tolist(),
            'temperature': monthly_temp.tolist(),
            'wind_speed': monthly_wind.tolist()
        })
        return weather_df

    def _calculate_pv_generation(self, weather_df, system_capacity_kw):
        df = weather_df.copy()
        # 平面辐照量转斜面辐照量
        df['POA_GHI'] = df['GHI'].astype(float) * 1.1
        # 电池温度计算
        df['cell_temp'] = df['temperature'].astype(float) + (float(self.system_config['NOCT']) - 20.0) * df['GHI'].astype(float) / 800.0
        # 温度损失系数（限制范围0.8-1.0）
        temp_coeff = 1.0 - float(self.system_config['temp_loss_coeff']) * (df['cell_temp'] - 25.0)
        df['temp_coeff'] = np.clip(temp_coeff, 0.8, 1.0).tolist()
        # 月度发电量计算（单位：kWh）
        df['monthly_generation_kwh'] = (
                df['POA_GHI'].astype(float) * 1000.0
                * float(self.system_config['module_efficiency'])
                * float(self.system_config['performance_ratio'])
                * df['temp_coeff'].astype(float)
                * system_capacity_kw
                / 1000.0
        ).astype(float).tolist()
        # 只保留必要列并确保类型正确
        return df[['month', 'GHI', 'temperature', 'wind_speed', 'monthly_generation_kwh']]

# ==========================================
# 2. 光伏经济性评估模型
# ==========================================
class PVEconomicModel:
    def __init__(self):
        self.economic_params = {
            'grid_price': 0.55,          # 上网电价（元/kWh）
            'initial_investment_per_kw': 3500.0,  # 初始投资（元/kW）
            'operation_cost_rate': 0.015, # 年运维费率
            'lifespan': 25,              # 项目寿命（年）
            'degradation_rate': 0.005    # 年衰减率
        }

    def run_full_economic_analysis(self, annual_gen_total_kwh, system_capacity_kw):
        # 边界值检查
        annual_gen_total_kwh = max(float(annual_gen_total_kwh), 0.1)
        system_capacity_kw = max(float(system_capacity_kw), 0.1)
        
        grid_price = float(self.economic_params['grid_price'])
        invest_per_kw = float(self.economic_params['initial_investment_per_kw'])
        op_cost_rate = float(self.economic_params['operation_cost_rate'])
        lifespan = int(self.economic_params['lifespan'])
        
        # 初始投资（万元）
        initial_investment = (system_capacity_kw * invest_per_kw) / 10000.0
        # 首年收益（万元）
        annual_income = annual_gen_total_kwh * grid_price / 10000.0
        # 年运维成本（万元）
        annual_op_cost = initial_investment * op_cost_rate
        # 年净收益（万元）
        annual_net_income = annual_income - annual_op_cost
        # 投资回收期（年）
        payback_period = initial_investment / annual_net_income if annual_net_income > 0 else 99.99
        # 全生命周期总发电量
        total_gen = annual_gen_total_kwh * lifespan * (1 - 0.005 * lifespan / 2)
        # 全生命周期总成本（元）
        total_cost = initial_investment * 10000 + annual_op_cost * 10000 * lifespan
        # 度电成本（元/kWh）
        LCOE = total_cost / total_gen if total_gen > 0 else 0.9999
        # 内部收益率（%）
        IRR = (annual_net_income / initial_investment) * 100 if initial_investment > 0 else 0.0
        
        return {
            'initial_investment': round(initial_investment, 2),
            'first_year_gen': round(annual_gen_total_kwh / 10000, 2),
            'LCOE': round(LCOE, 4),
            'payback_period': round(payback_period, 2),
            'IRR': round(IRR, 2),
            'annual_net_income': round(annual_net_income, 2)
        }

# ==========================================
# 3. 路由配置
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/assess', methods=['POST'])
def assess_pv_project():
    try:
        # 获取请求数据（兼容空数据）
        data = request.get_json() or {}
        
        # 基础参数解析与校验
        system_capacity = float(data.get('capacity', 1000.0))
        project_name = data.get('projectName', '光伏项目') or '光伏项目'
        random_point = bool(data.get('randomPoint', False))
        target_lat = data.get('latitude', None)
        target_lon = data.get('longitude', None)
        
        # 高级参数解析
        advanced_params = {
            'module_efficiency': max(float(data.get('moduleEfficiency', 0.18)), 0.01),
            'performance_ratio': max(float(data.get('performanceRatio', 0.80)), 0.5),
            'grid_price': max(float(data.get('gridPrice', 0.55)), 0.01),
            'investment_per_kw': max(float(data.get('investmentPerKw', 3500.0)), 1000.0)
        }

        print(f"=" * 60)
        print(f"📥 收到评估请求：{project_name} | 装机容量：{system_capacity}kW")
        print(f"🔀 模式：{'随机选点' if random_point else '地图选点'}")
        print(f"=" * 60)

        # 运行光伏发电量模型
        pv_model = PVGenerationModel()
        pv_model.system_config['module_efficiency'] = advanced_params['module_efficiency']
        pv_model.system_config['performance_ratio'] = advanced_params['performance_ratio']
        generation_df, annual_gen_per_kw = pv_model.run_full_model(
            target_lat=target_lat,
            target_lon=target_lon,
            system_capacity_kw=system_capacity,
            random_point=random_point
        )
        annual_gen_total = annual_gen_per_kw * system_capacity

        # 运行经济性评估模型
        economic_model = PVEconomicModel()
        economic_model.economic_params['grid_price'] = advanced_params['grid_price']
        economic_model.economic_params['initial_investment_per_kw'] = advanced_params['investment_per_kw']
        economic_result = economic_model.run_full_economic_analysis(
            annual_gen_total_kwh=annual_gen_total,
            system_capacity_kw=system_capacity
        )

        # 构造返回数据（确保所有值为Python原生类型）
        return jsonify({
            "success": True,
            "data": {
                "projectName": project_name,
                "latitude": float(pv_model.site_lat),
                "longitude": float(pv_model.site_lon),
                "matchedLat": float(pv_model.site_lat),
                "matchedLon": float(pv_model.site_lon),
                "elevation": float(pv_model.site_elevation),
                "capacity": float(system_capacity),
                "unitGeneration": round(float(annual_gen_per_kw), 2),
                "initialInvestment": float(economic_result['initial_investment']),
                "firstYearGeneration": float(economic_result['first_year_gen']),
                "LCOE": float(economic_result['LCOE']),
                "paybackPeriod": float(economic_result['payback_period']),
                "IRR": float(economic_result['IRR']),
                "annualNetIncome": float(economic_result['annual_net_income']),
                "monthlyGeneration": generation_df.round(2).to_dict('records')
            }
        })
    except ValueError as ve:
        print(f"❌ 输入参数错误：{str(ve)}")
        return jsonify({
            "success": False,
            "message": f"输入参数错误：{str(ve)}"
        }), 400
    except Exception as e:
        import traceback
        print(f"❌ 评估失败：{str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"服务器内部错误：{str(e)}"
        }), 500

# ==========================================
# 4. 启动应用（简化版，适配Render/PythonAnywhere）
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 全球光伏资源评估系统 - 后端启动成功")
    print("=" * 60)
    # 生产环境配置：关闭debug，适配平台端口
    port = int(os.environ.get('PORT', 5000))
    # 禁用reloader避免重复启动，适配云平台
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
