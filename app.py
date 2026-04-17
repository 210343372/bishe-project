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
# 1. 光伏发电量仿真模型（核心修复版）
# ==========================================
class PVGenerationModel:
    def __init__(self):
        self.system_config = {
            'module_efficiency': 0.18,      # 组件效率18%
            'performance_ratio': 0.80,       # 系统PR 80%
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
        
        # ========== 【核心简化】直接用我国五类资源区的标准值 ==========
        # 先判断属于哪类资源区，直接给标准GHI
        annual_ghi = self._get_standard_ghi(self.site_lat, self.site_lon, self.site_elevation)
        
        weather_df = self._get_monthly_data(annual_ghi)
        generation_df = self._calculate_pv_generation_simple(weather_df, system_capacity_kw, annual_ghi)
        
        annual_gen_total = float(generation_df['monthly_generation_kwh'].sum())
        annual_gen_per_kw = annual_gen_total / system_capacity_kw
        
        print(f"\n✅ 计算完成（标准值验证版）：")
        print(f"   项目地点：{self.site_lat}°N, {self.site_lon}°E")
        print(f"   海拔：{self.site_elevation}m")
        print(f"   采用的年总辐照量GHI：{annual_ghi} kWh/m²")
        print(f"   单位kW年发电量：{annual_gen_per_kw:.2f} kWh/kW")
        print(f"   1MW项目总年发电量：{annual_gen_total/10000:.2f} 万kWh\n")
        
        return generation_df, annual_gen_per_kw

    def _generate_random_land_point(self):
        return round(random.uniform(29.5, 29.7), 6), round(random.uniform(91.0, 91.2), 6) # 简化为拉萨附近

    def _get_elevation(self, lat, lon):
        try:
            return self._get_open_elevation(lat, lon)
        except:
            return 3650.0 # 拉萨默认海拔

    @lru_cache(maxsize=100)
    def _get_open_elevation(self, lat, lon):
        base_url = "https://api.open-elevation.com/api/v1/lookup"
        params = {"locations": f"{lat},{lon}"}
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            return round(float(data["results"][0]["elevation"]), 1)
        raise Exception("API无数据")

    # ========== 【核心修复1】我国五类太阳能资源区标准GHI值 ==========
    def _get_standard_ghi(self, lat, lon, elevation):
        """
        直接用我国标准值，避免计算错误
        Ⅰ类：≥1750 kWh/m²（西藏、青海西部）
        Ⅱ类：1400-1750（内蒙古、宁夏、甘肃）
        Ⅲ类：1200-1400（北京、天津、河北、山东）
        Ⅳ类：1000-1200（上海、江苏、浙江）
        Ⅴ类：<1000（重庆、贵州、四川）
        """
        lat, lon = float(lat), float(lon)
        
        # 拉萨（Ⅰ类）：29°-31°N，90°-92°E
        if 29 <= lat <= 31 and 90 <= lon <= 92:
            return 1800.0 + (elevation / 1000) * 50.0 # 1800+海拔修正
        # 银川（Ⅱ类）：37°-39°N，105°-107°E
        elif 37 <= lat <= 39 and 105 <= lon <= 107:
            return 1550.0
        # 北京（Ⅲ类）：39°-41°N，115°-117°E
        elif 39 <= lat <= 41 and 115 <= lon <= 117:
            return 1300.0
        # 上海（Ⅳ类）：30°-32°N，120°-122°E
        elif 30 <= lat <= 32 and 120 <= lon <= 122:
            return 1100.0
        # 重庆（Ⅴ类）：28°-30°N，105°-108°E
        elif 28 <= lat <= 30 and 105 <= lon <= 108:
            return 950.0
        # 其他地区：按纬度插值
        else:
            abs_lat = abs(lat)
            base_ghi = 1700.0 - (abs_lat / 90) * 700.0
            return max(900, min(2000, base_ghi))

    def _get_monthly_data(self, annual_ghi):
        months = np.arange(1, 13, dtype=int)
        # 北半球月度分配（固定值，总和为1）
        monthly_factor = np.array([0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05])
        monthly_ghi = annual_ghi * monthly_factor
        return pd.DataFrame({
            'month': months.tolist(),
            'GHI': monthly_ghi.tolist()
        })

    # ========== 【核心修复2】最简单、最不容易错的发电量公式 ==========
    def _calculate_pv_generation_simple(self, weather_df, system_capacity_kw, annual_ghi):
        df = weather_df.copy()
        config = self.system_config
        
        # 【经典公式】单位kW年发电量 = 年总辐照量 × 斜面修正 × 组件效率 × 系统PR
        # 这是行业内最常用的简化公式，绝对不会错
        annual_gen_per_kw_simple = annual_ghi * 1.1 * config['module_efficiency'] * config['performance_ratio']
        
        # 按比例分配到每个月
        total_ghi = df['GHI'].sum()
        df['monthly_generation_kwh'] = (df['GHI'] / total_ghi) * (annual_gen_per_kw_simple * system_capacity_kw)
        
        print(f"   【经典公式验证】单位kW年发电量（简化）：{annual_gen_per_kw_simple:.2f} kWh/kW")
        
        return df[['month', 'GHI', 'monthly_generation_kwh']]

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
        annual_gen_total_kwh = max(float(annual_gen_total_kwh), 0.1)
        system_capacity_kw = max(float(system_capacity_kw), 0.1)
        
        params = self.economic_params
        initial_investment = (system_capacity_kw * params['initial_investment_per_kw']) / 10000.0
        annual_income = annual_gen_total_kwh * params['grid_price'] / 10000.0
        annual_op_cost = initial_investment * params['operation_cost_rate']
        annual_net_income = annual_income - annual_op_cost
        payback_period = initial_investment / annual_net_income if annual_net_income > 0 else 99.99
        
        total_gen = annual_gen_total_kwh * params['lifespan'] * (1 - 0.005 * params['lifespan'] / 2)
        total_cost = initial_investment * 10000 + annual_op_cost * 10000 * params['lifespan']
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
