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
            'module_efficiency': 0.18,      # 组件效率
            'performance_ratio': 0.80,       # 系统性能比
            'NOCT': 45,                       # 标称工作温度
            'temp_loss_coeff': 0.004          # 温度损失系数
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
        weather_df = self._get_local_weather_data()
        generation_df = self._calculate_pv_generation(weather_df, system_capacity_kw)
        
        annual_gen_total = float(generation_df['monthly_generation_kwh'].sum())
        annual_gen_per_kw = annual_gen_total / system_capacity_kw
        
        print(f"\n✅ 计算完成：")
        print(f"   年总发电量：{annual_gen_total/10000:.2f} 万kWh")
        print(f"   单位kW年发电量：{annual_gen_per_kw:.2f} kWh/kW\n")
        
        return generation_df, annual_gen_per_kw

    def _generate_random_land_point(self):
        land_regions = [(20.0, 50.0, 100.0, 140.0)] # 简化为中国区域
        region = random.choice(land_regions)
        lat_min, lat_max, lon_min, lon_max = region
        return round(random.uniform(lat_min, lat_max), 6), round(random.uniform(lon_min, lon_max), 6)

    def _get_elevation(self, lat, lon):
        # 优先用免费的Open-Elevation API，和高德Key无关
        try:
            return self._get_open_elevation(lat, lon)
        except:
            return self._get_simulation_elevation(lat, lon)

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

    def _get_simulation_elevation(self, lat, lon):
        lat, lon = float(lat), float(lon)
        if 25 <= lat <= 35 and 75 <= lon <= 105:
            return round(np.random.uniform(2000, 4500), 1) # 青藏高原
        elif 35 <= lat <= 45 and 100 <= lon <= 120:
            return round(np.random.uniform(500, 1500), 1) # 北方高原
        else:
            return round(np.random.uniform(10, 500), 1) # 东部平原

    def _get_local_weather_data(self):
        abs_lat = float(abs(self.site_lat))
        months = np.arange(1, 13, dtype=int)
        
        # ========== 1. 年总辐照量GHI计算（修正版） ==========
        # 基础GHI：纬度越高，辐照越低（符合我国资源区分布）
        base_ghi = 1700.0 - (abs_lat / 90) * 700.0
        # 海拔修正：每升高1000m，辐照增加5%
        elevation_correction = (self.site_elevation / 1000) * 0.05 * base_ghi
        annual_ghi = base_ghi + elevation_correction
        # 范围限制：符合我国五类资源区
        annual_ghi = max(900.0, min(2000.0, annual_ghi))
        
        # ========== 2. 月度辐照量分配（修正版，体现南北差异） ==========
        if self.site_lat > 0: # 北半球
            monthly_factor = np.array([0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05])
        else:
            monthly_factor = np.array([0.14, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.05, 0.06, 0.08, 0.10, 0.12])
        
        monthly_ghi = annual_ghi * monthly_factor
        
        # ========== 3. 温度计算 ==========
        base_temp = 20.0 - (abs_lat / 90) * 25.0
        elevation_temp_correction = (self.site_elevation / 1000) * (-6.0)
        monthly_temp = base_temp + elevation_temp_correction + 12 * np.sin(np.radians((months - 3) * 30))
        
        return pd.DataFrame({
            'month': months.tolist(),
            'GHI': monthly_ghi.tolist(), # 月度总辐照量，kWh/m²
            'temperature': monthly_temp.tolist(),
            'days_in_month': [31,28,31,30,31,30,31,31,30,31,30,31] # 新增：每月天数
        })

    def _calculate_pv_generation(self, weather_df, system_capacity_kw):
        df = weather_df.copy()
        config = self.system_config
        
        # ========== 核心修复1：NOCT温度模型（用日平均辐照量，不是月度总辐照量） ==========
        # 日平均辐照量 = 月度总辐照量 / 当月天数，单位 kWh/m²/天
        daily_ghi = df['GHI'] / df['days_in_month']
        # 转换为平均辐照强度 W/m² (1kWh/m²/天 = 1000Wh / 24h ≈ 41.67 W/m²)
        avg_irradiance = daily_ghi * 1000 / 24
        
        # 计算组件温度（用平均辐照强度）
        df['cell_temp'] = df['temperature'] + (config['NOCT'] - 20.0) * (avg_irradiance / 800.0)
        
        # 计算温度损失系数
        temp_coeff = 1.0 - config['temp_loss_coeff'] * (df['cell_temp'] - 25.0)
        df['temp_coeff'] = np.clip(temp_coeff, 0.85, 1.0).tolist()
        
        # ========== 核心修复2：发电量计算公式（单位完全正确） ==========
        # 正确公式：
        # 月度发电量(kWh) = 月度总辐照量(kWh/m²) × 斜面修正(1.1) × 组件效率 × 系统PR × 温度损失 × 装机容量(kW)
        # 单位逻辑：kWh/m² × kW = kWh (因为1kW装机在1kWh/m²辐照下，理论发1kWh电，再乘以效率)
        df['monthly_generation_kwh'] = (
                df['GHI'] * 1.1
                * config['module_efficiency']
                * config['performance_ratio']
                * df['temp_coeff']
                * system_capacity_kw
        ).tolist()
        
        return df[['month', 'GHI', 'temperature', 'monthly_generation_kwh']]

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
