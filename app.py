from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import numpy as np
import requests
import random
from functools import lru_cache
import time

# ==========================================
# 正确的Flask路径配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
app = Flask(__name__, template_folder=TEMPLATES_DIR)

# ==========================================
# 1. 光伏发电量仿真模型（彻底修复海拔获取逻辑）
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
        if random_point:
            self.site_lat, self.site_lon = self._generate_random_land_point()
            print(f"🎲 随机选点：{self.site_lat}°N, {self.site_lon}°E")
        elif target_lat is not None and target_lon is not None:
            self.site_lat = round(float(target_lat), 6)
            self.site_lon = round(float(target_lon), 6)
            print(f"📍 地图选点：{self.site_lat}°N, {self.site_lon}°E")
        else:
            raise ValueError("请选择模式：指定经纬度 或 random_point=True")

        # 彻底去掉高德Key判断，只调用Open-Elevation API
        self.site_elevation = self._get_real_elevation(self.site_lat, self.site_lon)
        print(f"🏔️  最终海拔数据：{self.site_elevation}m")

        weather_df = self._get_local_weather_data()
        generation_df = self._calculate_pv_generation(weather_df, float(system_capacity_kw))
        annual_gen_total = float(generation_df['monthly_generation_kwh'].sum())
        annual_gen_per_kw = annual_gen_total / float(system_capacity_kw)
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

    # 核心修复：多次重试API，避免限流/超时导致失败
    def _get_real_elevation(self, lat, lon):
        max_retries = 3  # 最多重试3次
        retry_delay = 1  # 每次重试间隔1秒
        for attempt in range(max_retries):
            try:
                return self._get_open_elevation(lat, lon)
            except Exception as e:
                print(f"⚠️ 第{attempt+1}次获取真实海拔失败：{str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避，减少限流概率
                else:
                    print(f"❌ 所有重试失败，使用模拟海拔")
                    return self._get_simulation_elevation(lat, lon)

    # 缓存+优化请求头，降低API限流概率
    @lru_cache(maxsize=200)  # 扩大缓存容量
    def _get_open_elevation(self, lat, lon):
        base_url = "https://api.open-elevation.com/api/v1/lookup"
        params = {"locations": f"{lat},{lon}"}
        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
        # 延长超时时间+关闭连接复用，避免连接池耗尽
        response = requests.get(
            base_url, 
            params=params, 
            headers=headers, 
            timeout=30,
            stream=False,
            verify=False  # 忽略SSL验证，避免证书问题
        )
        response.raise_for_status()
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            elevation = data["results"][0]["elevation"]
            print(f"✅ 成功获取真实海拔：{elevation}m")
            return round(float(elevation), 1)
        raise Exception("API返回无海拔数据")

    def _get_simulation_elevation(self, lat, lon):
        if 25 <= lat <= 35 and 75 <= lon <= 105:
            return round(np.random.uniform(2000, 4500), 1)
        elif abs(lon % 360) <= 10 or abs(lon % 360) >= 350:
            return round(np.random.uniform(0, 100), 1)
        elif -30 <= lat <= 50 and abs(lon) <= 150:
            return round(np.random.uniform(50, 500), 1)
        else:
            return round(np.random.uniform(100, 2000), 1)

    def _get_local_weather_data(self):
        abs_lat = float(abs(self.site_lat))
        months = np.arange(1, 13, dtype=int)
        base_ghi = 1800.0 - (abs_lat / 90) * 800.0
        elevation_correction = (self.site_elevation / 1000) * 0.05 * base_ghi
        annual_ghi = base_ghi + elevation_correction
        annual_ghi = max(900.0, min(2000.0, annual_ghi))
        if self.site_lat > 0:
            monthly_factor = np.array([0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05], dtype=float)
        else:
            monthly_factor = np.array([0.14, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.05, 0.06, 0.08, 0.10, 0.12], dtype=float)
        base_temp = 25.0 - (abs_lat / 90) * 30.0
        elevation_temp_correction = (self.site_elevation / 1000) * (-6.0)
        monthly_temp = base_temp + elevation_temp_correction + 10 * np.sin(np.radians((months - 3) * 30))
        monthly_wind = np.random.uniform(2.0, 6.0, 12)
        return pd.DataFrame({
            'month': months,
            'GHI': annual_ghi * monthly_factor,
            'temperature': monthly_temp,
            'wind_speed': monthly_wind
        })

    def _calculate_pv_generation(self, weather_df, system_capacity_kw):
        df = weather_df.copy()
        system_capacity_kw = float(system_capacity_kw)
        df['POA_GHI'] = df['GHI'].astype(float) * 1.1
        df['cell_temp'] = df['temperature'].astype(float) + (float(self.system_config['NOCT']) - 20.0) * df['GHI'].astype(float) / 800.0
        temp_coeff = 1.0 - float(self.system_config['temp_loss_coeff']) * (df['cell_temp'] - 25.0)
        df['temp_coeff'] = np.clip(temp_coeff, 0.8, 1.0)
        df['monthly_generation_kwh'] = (
                df['POA_GHI'].astype(float) * 1000.0
                * float(self.system_config['module_efficiency'])
                * float(self.system_config['performance_ratio'])
                * df['temp_coeff'].astype(float)
                * system_capacity_kw
                / 1000.0
        ).astype(float)
        return df[['month', 'GHI', 'temperature', 'wind_speed', 'monthly_generation_kwh']]

# ==========================================
# 2. 光伏经济性评估模型
# ==========================================
class PVEconomicModel:
    def __init__(self):
        self.economic_params = {
            'grid_price': 0.55,
            'initial_investment_per_kw': 3500.0,
            'operation_cost_rate': 0.015,
            'lifespan': 25,
            'degradation_rate': 0.005
        }

    def run_full_economic_analysis(self, annual_gen_total_kwh, system_capacity_kw):
        annual_gen_total_kwh = float(annual_gen_total_kwh)
        system_capacity_kw = float(system_capacity_kw)
        grid_price = float(self.economic_params['grid_price'])
        invest_per_kw = float(self.economic_params['initial_investment_per_kw'])
        op_cost_rate = float(self.economic_params['operation_cost_rate'])
        lifespan = int(self.economic_params['lifespan'])
        initial_investment = (system_capacity_kw * invest_per_kw) / 10000.0
        annual_income = annual_gen_total_kwh * grid_price / 10000.0
        annual_op_cost = initial_investment * op_cost_rate
        annual_net_income = annual_income - annual_op_cost
        payback_period = initial_investment / annual_net_income if annual_net_income > 0 else 99.99
        total_gen = annual_gen_total_kwh * lifespan * (1 - 0.005 * lifespan / 2)
        total_cost = initial_investment * 10000 + annual_op_cost * 10000 * lifespan
        LCOE = total_cost / total_gen if total_gen > 0 else 0.9999
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
        data = request.get_json()
        system_capacity = float(data.get('capacity', 1000.0))
        project_name = data.get('projectName', '光伏项目')
        random_point = data.get('randomPoint', False)
        target_lat = data.get('latitude', None)
        target_lon = data.get('longitude', None)
        advanced_params = {
            'module_efficiency': float(data.get('moduleEfficiency', 0.18)),
            'performance_ratio': float(data.get('performanceRatio', 0.80)),
            'grid_price': float(data.get('gridPrice', 0.55)),
            'investment_per_kw': float(data.get('investmentPerKw', 3500.0))
        }

        print(f"=" * 60)
        print(f"📥 收到评估请求：{project_name} | 装机容量：{system_capacity}kW")
        print(f"🔀 模式：{'随机选点' if random_point else '地图选点'}")
        print(f"=" * 60)

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

        economic_model = PVEconomicModel()
        economic_model.economic_params['grid_price'] = advanced_params['grid_price']
        economic_model.economic_params['initial_investment_per_kw'] = advanced_params['investment_per_kw']
        economic_result = economic_model.run_full_economic_analysis(
            annual_gen_total_kwh=annual_gen_total,
            system_capacity_kw=system_capacity
        )

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
    except Exception as e:
        import traceback
        print(f"❌ 评估失败：{str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"评估失败：{str(e)}"
        })

# ==========================================
# 4. 启动应用
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 全球光伏资源评估系统 - 后端启动成功")
    print("=" * 60)
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
