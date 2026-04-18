from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import numpy as np
import requests
import random
from functools import lru_cache

# ==========================================
# Flask路径配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
app = Flask(__name__, template_folder=TEMPLATES_DIR)

# ==========================================
# 配置
# ==========================================
AMAP_KEY = "44d3e38673dc18b760f544a0d48f8f7f"

# ==========================================
# 1. 光伏发电量仿真模型（最终修正版）
# ==========================================
class PVGenerationModel:
    def __init__(self):
        self.system_config = {
            'module_efficiency': 0.18,      # 仅做展示，不参与核心计算
            'performance_ratio': 0.80,       # 系统性能比PR，行业通用0.75-0.85
            'NOCT': 45,
            'temp_loss_coeff': 0.004
        }
        self.site_lat = None
        self.site_lon = None
        self.site_elevation = None

    def run_full_model(self, target_lat=None, target_lon=None, system_capacity_kw=1, random_point=False):
        system_capacity_kw = max(float(system_capacity_kw), 0.1)
        
        if random_point:
            self.site_lat, self.site_lon = self._generate_random_land_point()
        elif target_lat is not None and target_lon is not None:
            self.site_lat = round(float(target_lat), 6)
            self.site_lon = round(float(target_lon), 6)
        else:
            raise ValueError("请指定经纬度或使用随机选点")

        self.site_elevation = self._get_elevation(self.site_lat, self.site_lon)
        
        # 获取标准年总辐照量GHI
        annual_ghi = self._get_standard_ghi(self.site_lat, self.site_lon, self.site_elevation)
        
        weather_df = self._get_monthly_data(annual_ghi)
        generation_df, annual_gen_per_kw = self._calculate_pv_generation_correct(weather_df, system_capacity_kw, annual_ghi)
        
        annual_gen_total = annual_gen_per_kw * system_capacity_kw
        
        print(f"\n✅ 计算完成（行业标准公式版）：")
        print(f"   项目地点：{self.site_lat}°N, {self.site_lon}°E")
        print(f"   海拔：{self.site_elevation}m")
        print(f"   采用的年总辐照量GHI：{annual_ghi} kWh/m²")
        print(f"   单位kW年发电量：{annual_gen_per_kw:.2f} kWh/kW")
        print(f"   {system_capacity_kw}kW项目总年发电量：{annual_gen_total/10000:.2f} 万kWh\n")
        
        return generation_df, annual_gen_per_kw

    def _generate_random_land_point(self):
        # 中国区域随机选点
        lat_min, lat_max = 20.0, 50.0
        lon_min, lon_max = 73.0, 135.0
        return round(random.uniform(lat_min, lat_max), 6), round(random.uniform(lon_min, lon_max), 6)

    def _get_elevation(self, lat, lon):
        try:
            return self._get_open_elevation(lat, lon)
        except:
            # 模拟海拔兜底
            lat, lon = float(lat), float(lon)
            if 29 <= lat <= 31 and 90 <= lon <= 92:
                return 3650.0
            elif 37 <= lat <= 39 and 105 <= lon <= 107:
                return 1110.0
            elif 39 <= lat <= 41 and 115 <= lon <= 117:
                return 43.0
            elif 30 <= lat <= 32 and 120 <= lon <= 122:
                return 4.0
            elif 28 <= lat <= 30 and 105 <= lon <= 108:
                return 259.0
            else:
                return 100.0

    @lru_cache(maxsize=100)
    def _get_open_elevation(self, lat, lon):
        base_url = "https://api.open-elevation.com/api/v1/lookup"
        params = {"locations": f"{lat},{lon}"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            elevation = data["results"][0]["elevation"]
            return round(float(elevation), 1)
        raise Exception("API无有效海拔数据")

    # 我国五类太阳能资源区标准GHI值（国标GB/T 37526-2019）
    def _get_standard_ghi(self, lat, lon, elevation):
        lat, lon = float(lat), float(lon)
        
        # 拉萨（Ⅰ类资源区）：29°-31°N，90°-92°E
        if 29 <= lat <= 31 and 90 <= lon <= 92:
            base_ghi = 1800.0
            # 海拔修正：每升高1000m，辐照增加5%
            elevation_correction = (elevation / 1000) * 0.05 * base_ghi
            return base_ghi + elevation_correction
        # 银川（Ⅱ类资源区）：37°-39°N，105°-107°E
        elif 37 <= lat <= 39 and 105 <= lon <= 107:
            return 1550.0
        # 北京（Ⅲ类资源区）：39°-41°N，115°-117°E
        elif 39 <= lat <= 41 and 115 <= lon <= 117:
            return 1300.0
        # 上海（Ⅳ类资源区）：30°-32°N，120°-122°E
        elif 30 <= lat <= 32 and 120 <= lon <= 122:
            return 1100.0
        # 重庆（Ⅴ类资源区）：28°-30°N，105°-108°E
        elif 28 <= lat <= 30 and 105 <= lon <= 108:
            return 950.0
        # 其他地区：按纬度插值，符合我国分布
        else:
            abs_lat = abs(lat)
            base_ghi = 1700.0 - (abs_lat / 90) * 700.0
            # 海拔修正
            elevation_correction = (elevation / 1000) * 0.05 * base_ghi
            return max(900.0, min(2000.0, base_ghi + elevation_correction))

    def _get_monthly_data(self, annual_ghi):
        months = np.arange(1, 13, dtype=int)
        # 北半球月度辐照分配（行业通用，总和为1）
        if self.site_lat >= 0:
            monthly_factor = np.array([0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05])
        else:
            monthly_factor = np.array([0.14, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.05, 0.06, 0.08, 0.10, 0.12])
        monthly_ghi = annual_ghi * monthly_factor
        return pd.DataFrame({
            'month': months.tolist(),
            'GHI': monthly_ghi.tolist()
        })

    # ========== 【最终修正版】行业标准发电量计算公式 ==========
    def _calculate_pv_generation_correct(self, weather_df, system_capacity_kw, annual_ghi):
        df = weather_df.copy()
        config = self.system_config
        
        # 【核心正确公式】单位kW年发电量 = 年辐照量 × 斜面修正系数 × 系统性能比PR
        # 1kWp装机已经包含组件效率，无需重复相乘！
        slope_correction = 1.1  # 最佳倾角斜面修正，行业通用值
        annual_gen_per_kw = annual_ghi * slope_correction * config['performance_ratio']
        
        # 按月度辐照占比，分配月度发电量
        total_ghi = df['GHI'].sum()
        df['monthly_generation_kwh'] = (df['GHI'] / total_ghi) * (annual_gen_per_kw * system_capacity_kw)
        
        # 打印验证，确保公式正确
        print(f"   【公式验证】{annual_ghi} kWh/m² × {slope_correction} × {config['performance_ratio']} = {annual_gen_per_kw:.2f} kWh/kW")
        
        return df[['month', 'GHI', 'monthly_generation_kwh']], annual_gen_per_kw

# ==========================================
# 2. 光伏经济性评估模型（行业标准修正版）
# ==========================================
class PVEconomicModel:
    def __init__(self):
        self.economic_params = {
            'grid_price': 0.55,                # 上网电价（元/kWh）
            'initial_investment_per_kw': 3500.0,# 单位千瓦初始投资（元/kW）
            'operation_cost_rate': 0.015,       # 首年运维费率（初始投资占比）
            'op_cost_growth_rate': 0.02,        # 运维成本年增长率（行业通用2%）
            'lifespan': 25,                      # 项目全生命周期（年）
            'first_year_degradation': 0.025,    # 组件首年衰减率（行业标准2.5%）
            'annual_degradation': 0.005,        # 组件次年起年衰减率（行业标准0.5%）
            'discount_rate': 0.06                # 基准折现率（国家规范推荐6%）
        }

    def run_full_economic_analysis(self, annual_gen_total_kwh, system_capacity_kw):
        # 边界值校验，避免除以0错误
        annual_gen_total_kwh = max(float(annual_gen_total_kwh), 0.1)
        system_capacity_kw = max(float(system_capacity_kw), 0.1)
        params = self.economic_params

        # ========== 1. 基础参数计算 ==========
        # 初始投资（万元）
        initial_investment = (system_capacity_kw * params['initial_investment_per_kw']) / 10000.0
        # 首年发电毛收益（万元）
        annual_income = annual_gen_total_kwh * params['grid_price'] / 10000.0
        # 首年运维成本（万元）
        annual_op_cost_first = initial_investment * params['operation_cost_rate']

        # ========== 2. 全生命周期现金流计算（核心修正） ==========
        # 初始化25年现金流列表，第0年为初始投资（负现金流）
        cash_flow = [-initial_investment]
        annual_gen_list = []
        annual_op_cost_list = []

        for year in range(1, params['lifespan'] + 1):
            # 计算当年发电量（考虑组件衰减）
            if year == 1:
                year_gen = annual_gen_total_kwh * (1 - params['first_year_degradation'])
            else:
                year_gen = annual_gen_list[-1] * (1 - params['annual_degradation'])
            annual_gen_list.append(year_gen)

            # 计算当年运维成本（考虑年增长）
            year_op_cost = annual_op_cost_first * (1 + params['op_cost_growth_rate']) ** (year - 1)
            annual_op_cost_list.append(year_op_cost)

            # 当年净现金流（万元）
            year_income = year_gen * params['grid_price'] / 10000.0
            year_net_cash = year_income - year_op_cost
            cash_flow.append(year_net_cash)

        # ========== 3. 核心指标计算（行业标准公式） ==========
        # 1. 首年净收益（万元）
        first_year_net_income = annual_income - annual_op_cost_first
        # 2. 静态投资回收期（年）
        static_payback = initial_investment / first_year_net_income if first_year_net_income > 0 else 99.99

        # 3. 平准化度电成本LCOE（行业标准折现法）
        total_cost_pv = initial_investment  # 折现总成本（初始投资第0年，无需折现）
        total_gen_pv = 0.0  # 折现总发电量
        for year in range(1, params['lifespan'] + 1):
            # 当年成本折现
            year_cost_pv = annual_op_cost_list[year-1] / ((1 + params['discount_rate']) ** year)
            total_cost_pv += year_cost_pv
            # 当年发电量折现
            year_gen_pv = annual_gen_list[year-1] / ((1 + params['discount_rate']) ** year)
            total_gen_pv += year_gen_pv
        LCOE = (total_cost_pv * 10000) / total_gen_pv if total_gen_pv > 0 else 0.9999

        # 4. 内部收益率IRR（牛顿迭代法求解动态IRR）
        def npv(irr):
            return sum(cf / ((1 + irr) ** t) for t, cf in enumerate(cash_flow))
        
        # 迭代求解IRR
        irr_low = 0.0
        irr_high = 0.5
        irr_result = 0.0
        # 迭代100次，精度0.0001
        for _ in range(100):
            irr_mid = (irr_low + irr_high) / 2
            npv_mid = npv(irr_mid)
            if abs(npv_mid) < 1e-4:
                irr_result = irr_mid
                break
            elif npv_mid > 0:
                irr_low = irr_mid
            else:
                irr_high = irr_mid
        IRR = irr_result * 100  # 转换为百分比

        # ========== 5. 返回结果（对齐PVsyst输出） ==========
        return {
            'initial_investment': round(initial_investment, 2),
            'first_year_gen': round(annual_gen_total_kwh / 10000, 2),
            'first_year_income': round(annual_income, 2),
            'first_year_op_cost': round(annual_op_cost_first, 2),
            'annual_net_income': round(first_year_net_income, 2),
            'payback_period': round(static_payback, 2),
            'LCOE': round(LCOE, 4),
            'IRR': round(IRR, 2)
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
        data = request.get_json() or {}
        system_capacity = float(data.get('capacity', 1000.0))
        project_name = data.get('projectName', '光伏项目') or '光伏项目'
        random_point = bool(data.get('randomPoint', False))
        target_lat = data.get('latitude', None)
        target_lon = data.get('longitude', None)
        
        advanced_params = {
            'module_efficiency': max(float(data.get('moduleEfficiency', 0.18)), 0.01),
            'performance_ratio': max(float(data.get('performanceRatio', 0.80)), 0.5),
            'grid_price': max(float(data.get('gridPrice', 0.55)), 0.01),
            'investment_per_kw': max(float(data.get('investmentPerKw', 3500.0)), 1000.0)
        }

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
        return jsonify({"success": False, "message": f"评估失败：{str(e)}"}), 500

# ==========================================
# 4. 启动应用
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 全球光伏资源评估系统 - 后端启动成功")
    print("=" * 60)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
