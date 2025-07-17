import pandas as pd
import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import make_interp_spline

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 空调系统热力学模型（改进版）
class ACSystemThermodynamics:
    def __init__(self, refrigerant="R134a", max_power=3000):
        self.refrigerant = refrigerant
        self.max_power = max_power  # 最大允许功率

    def get_enthalpy(self, T, P):
        """获取特定温度和压力下冷媒的焓值，增加更完善的错误处理"""
        try:
            h = CP.PropsSI('H', 'T', T + 273.15, 'P', P * 1e5, self.refrigerant)
            return h
        except:
            return 250000  # 典型焓值 J/kg

    def calculate_power(self, mass_flow_rate, T_in, P_in, T_out, P_out, work_ratio):
        """计算压缩机功率，增加功率限制"""
        h_in = self.get_enthalpy(T_in, P_in)
        h_out = self.get_enthalpy(T_out, P_out)

        # 确保焓差合理
        h_diff = max(30000, h_out - h_in)  # 最小焓差30kJ/kg

        power = mass_flow_rate * h_diff  # 单位：W
        power_with_ratio = power * work_ratio

        # 应用功率限制 - 提高最小功率到100W
        limited_power = max(100, min(power_with_ratio, self.max_power))
        return limited_power


# 空调系统能效计算（改进版）
class ACEfficiency:
    def __init__(self, cop_base=4.0, min_cop=2.0, max_cop=5.0):
        """初始化空调的COP，考虑COP随负载变化"""
        self.cop_base = cop_base
        self.min_cop = min_cop
        self.max_cop = max_cop

    def calculate_cop(self, power):
        load_ratio = power / 3000.0
        # 改进的变频特性曲线
        cop = 5.2 - 2.0 * load_ratio ** 2  # 低负载时COP可达5.2
        return np.clip(cop, 3.8, 5.2)  # 设置合理上下限

    def calculate_cooling_capacity(self, power):
        """根据功率和实际COP计算制冷能力 - 移除最大制冷量限制"""
        if power <= 0:
            return 0

        cop = self.calculate_cop(power)
        cooling_capacity = power * cop
        return cooling_capacity  # 移除3000W限制


# 控制算法（优化版PID）
class ACControl:
    def __init__(self, set_point=24.0, Kp=0.3, Ki=0.01, Kd=0.04):
        """初始化PID控制器，调整参数提高响应速度"""
        self.set_point = set_point
        self.pid = PID(Kp, Ki, Kd)

    def adjust_system(self, current_temp, outdoor_temp):
        """根据当前温度和室外温度调整控制信号"""
        # 增加室外温度对控制信号的影响
        temp_diff = outdoor_temp - current_temp
        control_signal = self.pid.compute(self.set_point, current_temp)

        # 室外温度越高，控制信号越大
        outdoor_factor = min(1.0, max(0.5, temp_diff / 10.0))
        adjusted_signal = control_signal * outdoor_factor

        return min(max(adjusted_signal, 0), 1.0)  # 限制在0-1范围内


class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0
        self.integral_limit = 1.5  # 调整积分限幅

    def compute(self, set_point, current_value):
        error = set_point - current_value

        # 改进的积分处理
        if abs(error) < 0.2:
            self.integral *= 0.9  # 接近目标时缓慢衰减积分
        else:
            self.integral += error

        # 积分限幅
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error

        return np.clip(output, 0, 1.2)  # 调整输出范围


# 节能控制器（优化版） - 增强启动过程
class EnergySavingController:
    def __init__(self, target_temp=24.0):
        """
        优化后的节能控制器（变频空调逻辑）
        主要改进：
        1. 动态死区控制
        2. PID信号整合
        3. 状态机严格管理
        4. 启动过程优化
        """
        self.target_temp = target_temp
        self.min_power_ratio = 0.02  # 基础功率降至2%（原5%）
        self.dead_band = 0.4  # 死区缩小到0.4°C（原0.8）
        self.max_power_ratio = 0.8  # 最大工作比例
        self.last_ratio = 0.0
        self.temp_history = []  # 温度记录窗口
        self.cooling_on = False  # 运行状态
        self.ramp_up_duration = 0.1  # 启动阶段缩短至6分钟（原15分钟）
        self.max_change = 0.03  # 功率变化率限制

        # PID补偿参数
        self.pid_gain = 0.2  # PID信号增益系数
        self.stable_threshold = 0.2  # 稳定区判定阈值

    def calculate_work_ratio(self, indoor_temp, outdoor_temp, elapsed_hours, pid_signal=0):
        """
        计算工作比例（整合PID控制）
        参数：
        - indoor_temp: 当前室内温度(°C)
        - outdoor_temp: 当前室外温度(°C)
        - elapsed_hours: 已运行时间(小时)
        - pid_signal: PID控制器输出信号(0-1)
        返回：
        - 工作比例(0-1)
        """
        # 记录温度历史（用于动态调节）
        self.temp_history.append(indoor_temp)
        if len(self.temp_history) > 10:
            self.temp_history.pop(0)

        # ===== 启动阶段处理 =====
        if elapsed_hours < self.ramp_up_duration:
            ramp_ratio = min(0.6, 0.3 + 0.5 * (elapsed_hours / self.ramp_up_duration))
            self.last_ratio = ramp_ratio
            self.cooling_on = True
            return ramp_ratio

        # ===== 动态死区调整 =====
        temp_deviation = indoor_temp - self.target_temp
        dynamic_dead_band = self._calculate_dynamic_dead_band()

        # ===== 核心状态机 =====
        if temp_deviation < -dynamic_dead_band:  # 过冷状态
            self.cooling_on = False
            target_ratio = 0.0

        elif temp_deviation > dynamic_dead_band:  # 过热状态
            self.cooling_on = True
            # 动态负载计算（含室外温度补偿）
            outdoor_comp = 0.05 * (outdoor_temp - 25) / 10
            base_ratio = 0.25 + 0.1 * min(temp_deviation, 3.0)  # 偏差增益
            target_ratio = min(base_ratio + outdoor_comp, self.max_power_ratio)

        else:  # 舒适区
            # PID信号补偿（仅在稳定区生效）
            pid_comp = pid_signal * self.pid_gain
            target_ratio = np.clip(self.min_power_ratio + pid_comp, 0, self.max_power_ratio)

        # ===== 功率平滑过渡 =====
        if target_ratio > self.last_ratio:
            new_ratio = min(self.last_ratio + self.max_change, target_ratio)
        else:
            new_ratio = max(self.last_ratio - self.max_change, target_ratio)

        self.last_ratio = new_ratio
        return new_ratio if self.cooling_on else 0.0

    def _calculate_dynamic_dead_band(self):
        """根据温度变化率动态调整死区"""
        if len(self.temp_history) < 3:
            return self.dead_band

        # 计算最近温度变化率 (°C/h)
        change_rate = abs(self.temp_history[-1] - self.temp_history[-3]) / 0.5

        # 变化率越大，死区越小（响应更快）
        dynamic_band = self.dead_band * (1 - 0.5 * min(change_rate, 1.0))
        return max(0.2, min(dynamic_band, 0.6))  # 限制在0.2-0.6°C


# 生成更自然的夏天温度曲线
def generate_summer_temperature(start_temp=28.3, duration_hours=24):
    """生成96个点（每15分钟一个点）的温度曲线"""
    # 生成96个时间点 (0, 0.25, 0.5, ..., 23.75)
    hours = np.linspace(0, duration_hours, 96)

    # 基础温度曲线
    base_temp = 30.0 + 7.0 * np.sin((hours - 6) * np.pi / 12)

    # 二级波动
    secondary_temp = 1.5 * np.sin(hours * np.pi / 8 + 0.5) + 0.8 * np.sin(hours * np.pi / 4)

    # 随机波动
    random_temp = np.random.normal(0, 0.4, len(hours))

    # 组合温度
    temperatures = base_temp + secondary_temp + random_temp
    temperatures = np.clip(temperatures, 28.0, 38.0)

    # 湿度和流量
    humidities = 60 - 0.8 * (temperatures - np.mean(temperatures)) + np.random.normal(0, 3, len(hours))
    humidities = np.clip(humidities, 40, 85)

    flows = 240 + 2.5 * (temperatures - np.min(temperatures)) + np.random.normal(0, 12, len(hours))
    flows = np.clip(flows, 200, 380)

    return pd.DataFrame({
        'hour': hours,
        'temperature': temperatures,
        'humidity': humidities,
        'flow': flows
    })
    return data


# 模拟空调系统的工作状态（优化版）
def simulate_ac_system_with_energy_saving(target_temp=24.0, env_data=None):  # 增加env_data参数
    # 使用传入的环境数据或生成新数据
    data = env_data if env_data is not None else generate_summer_temperature(start_temp=28.3, duration_hours=24)

    # 创建改进的系统组件
    thermodynamics = ACSystemThermodynamics(max_power=3000)  # 明确最大功率
    efficiency = ACEfficiency()
    control = ACControl(set_point=target_temp)
    energy_saver = EnergySavingController(target_temp=target_temp)

    # 初始化参数 - 调整房间热容和初始温差
    # 初始室内温度比室外高2度
    initial_indoor_temp = data.iloc[0]['temperature'] + 2.0
    indoor_temps = [initial_indoor_temp]
    cooling_capacities = []
    work_inputs = []
    work_ratios = []
    outdoor_temp_list = []
    control_signals = []
    solar_gains = []  # 记录太阳辐射热

    # 热力学参数
    C = 10.0e6  # 房间热容 (J/K)
    K = 35  # 热传导系数 (W/K)
    glass_area = 3.0  # 窗帘遮挡后有效玻璃面积 (平方米) - 原6㎡减半
    glass_transmittance = 0.7  # 玻璃透射率
    total_heat_sources = 1200  # 总热源 = 人类活动400W + 笔记本800W = 1200W
    dt = 900  # 时间步长 (15分钟)

    # 模拟过程
    elapsed_hours = 0.0
    for i in range(1, len(data)):
        current_outdoor_temp = data.iloc[i]['temperature']
        humidity = data.iloc[i]['humidity']
        flow = data.iloc[i]['flow']
        outdoor_temp_list.append(current_outdoor_temp)

        elapsed_hours += 0.25
        current_indoor_temp = indoor_temps[-1]

        # 计算工作比例
        work_ratio = energy_saver.calculate_work_ratio(
            current_indoor_temp,
            current_outdoor_temp,
            elapsed_hours
        )
        work_ratios.append(work_ratio)

        # 计算控制信号（加入室外温度）
        control_signal = control.adjust_system(current_indoor_temp, current_outdoor_temp)
        control_signals.append(control_signal)

        # 质量流量计算
        mass_flow_rate = flow * 0.0012  # 调整转换系数

        # 温度参数（改进温差计算）
        T_in = current_indoor_temp
        temp_deviation = current_indoor_temp - target_temp
        base_temp_diff = 1.8
        dynamic_adjust = min(1.2, max(0, abs(temp_deviation) * 0.6))
        temp_diff = base_temp_diff + dynamic_adjust
        T_out = T_in - temp_diff

        # 压力参数
        P_in = 2.2  # 调整进气压力
        P_out = 5.8  # 调整出气压力

        # 计算功率（应用工作比例）
        work_input = thermodynamics.calculate_power(
            mass_flow_rate, T_in, P_in, T_out, P_out, work_ratio
        )
        work_inputs.append(work_input)

        # 计算制冷能力
        cooling_capacity = efficiency.calculate_cooling_capacity(work_input)
        cooling_capacities.append(cooling_capacity)

        # 计算太阳辐射热 (动态模型)
        current_hour = elapsed_hours % 24
        # 太阳辐射强度 (W/m²) - 基于时间和室外温度
        if 6 <= current_hour < 18:
            # 正午12点达到峰值
            solar_intensity = 800 * np.sin(np.pi * (current_hour - 6) / 12)
            # 温度越高辐射越强
            solar_intensity *= (1 + 0.15 * (current_outdoor_temp - 28))
        else:
            solar_intensity = 0

        solar_gain = solar_intensity * glass_area * glass_transmittance
        solar_gains.append(solar_gain)

        # 计算下一时刻温度（改进热模型）
        heat_transfer = K * (current_outdoor_temp - current_indoor_temp)

        # 温度变化计算 - 包含太阳辐射热和所有热源
        dT = (heat_transfer + solar_gain + total_heat_sources - cooling_capacity) * dt / C
        new_indoor_temp = current_indoor_temp + dT

        # 温度合理性检查
        new_indoor_temp = max(18, min(new_indoor_temp, 35))  # 温度物理限制
        indoor_temps.append(new_indoor_temp)

    # 动态Y轴范围计算
    min_temp = min(min(data['temperature']), min(indoor_temps)) - 1
    max_temp = max(max(data['temperature']), max(indoor_temps)) + 1

    # 可视化结果
    plt.figure(figsize=(16, 12))

    # 温度变化图
    plt.subplot(3, 2, 1)
    plt.plot(data['hour'][1:], indoor_temps[1:], label="室内温度 (°C)", color='blue', linewidth=2)
    plt.plot(data['hour'], data['temperature'], label="室外温度 (°C)", color='red', alpha=0.7)
    plt.axhline(y=target_temp, color='green', linestyle='--', label="目标温度")
    plt.axhspan(target_temp - 0.5, target_temp + 0.5, color='lightgreen', alpha=0.2, label="舒适范围")
    plt.title(f"室内外温度变化 (目标温度: {target_temp}°C)", fontsize=14)
    plt.xlabel("时间 (小时)", fontsize=12)
    plt.ylabel("温度 (°C)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(min_temp, max_temp)

    # 功率变化图
    plt.subplot(3, 2, 2)
    plt.plot(data['hour'][1:], work_inputs, label="输入功率 (W)", color='orange')
    plt.axhline(y=3000, color='red', linestyle='--', label="最大功率限制")
    plt.title(f"空调输入功率 (目标温度: {target_temp}°C)", fontsize=14)
    plt.xlabel("时间 (小时)", fontsize=12)
    plt.ylabel("功率 (W)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 3300)  # 固定Y轴范围便于观察

    # 工作比例变化图
    plt.subplot(3, 2, 3)
    plt.plot(data['hour'][1:], work_ratios, label="工作比例", color='brown')
    plt.fill_between(data['hour'][1:], work_ratios, 0, color='brown', alpha=0.3)
    plt.title("节能控制器工作比例", fontsize=14)
    plt.xlabel("时间 (小时)", fontsize=12)
    plt.ylabel("工作比例 (0-1)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.0)

    # 累计能耗图
    plt.subplot(3, 2, 4)
    cumulative_energy = np.cumsum(work_inputs) * dt / 3600000  # 转换为kWh
    plt.plot(data['hour'][1:], cumulative_energy, label="累计能耗", color='darkred')
    plt.title(f"空调累计能耗 (目标温度: {target_temp}°C)", fontsize=14)
    plt.xlabel("时间 (小时)", fontsize=12)
    plt.ylabel("能耗 (kWh)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 太阳辐射热图
    plt.subplot(3, 2, 5)
    plt.plot(data['hour'][1:], solar_gains, label="太阳辐射热", color='gold')
    plt.title("太阳辐射热负荷 (3㎡玻璃窗 + 窗帘)", fontsize=14)
    plt.xlabel("时间 (小时)", fontsize=12)
    plt.ylabel("热负荷 (W)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 热源分解图
    plt.subplot(3, 2, 6)
    avg_heat_transfer = np.mean([K * (outdoor_temp_list[i] - indoor_temps[i]) for i in range(len(outdoor_temp_list))])
    heat_sources = [avg_heat_transfer, total_heat_sources, np.mean(solar_gains)]
    labels = ['热传导', '室内热源', '太阳辐射']

    plt.pie(heat_sources, labels=labels, autopct='%1.1f%%',
            colors=['lightblue', 'lightcoral', 'gold'], explode=(0.05, 0.05, 0.05))
    plt.title("热源分解 (平均)", fontsize=14)

    plt.tight_layout()
    plt.savefig(f'空调系统优化_{target_temp}°C_性能分析.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 计算并打印性能数据
    total_energy = cumulative_energy[-1] if len(cumulative_energy) > 0 else 0
    avg_work_ratio = np.mean(work_ratios) if work_ratios else 0
    avg_indoor_temp = np.mean(indoor_temps) if indoor_temps else 0
    temp_deviation = np.std(indoor_temps) if indoor_temps else 0
    max_power = max(work_inputs) if work_inputs else 0
    min_indoor_temp = min(indoor_temps) if indoor_temps else 0
    max_indoor_temp = max(indoor_temps) if indoor_temps else 0

    # 温度在目标范围内的比例
    lower_bound = target_temp - 0.5
    upper_bound = target_temp + 0.5
    in_target_range = sum(1 for t in indoor_temps if lower_bound <= t <= upper_bound) / len(indoor_temps) * 100

    print(f"\n目标温度 {target_temp}°C 优化后性能分析")
    print("=" * 60)
    print(f"总能耗: {total_energy:.3f} kWh")
    print(f"平均工作比例: {avg_work_ratio:.3f}")
    print(f"最大功率: {max_power:.1f} W (限制: 3000W)")
    print(f"室内温度范围: {min_indoor_temp:.1f}°C 至 {max_indoor_temp:.1f}°C")
    print(f"室内平均温度: {avg_indoor_temp:.3f} °C (目标: {target_temp}°C)")
    print(f"温度标准差: {temp_deviation:.3f} °C")
    print(f"温度在{lower_bound:.1f}-{upper_bound:.1f}°C范围内的比例: {in_target_range:.1f}%")
    print(f"平均COP: {np.mean([efficiency.calculate_cop(w) for w in work_inputs if w > 0]):.2f}")
    print(f"室内热源总量: {total_heat_sources} W (人类活动400W + 笔记本800W)")
    print(f"太阳辐射热(平均): {np.mean(solar_gains):.1f} W")
    print(f"窗帘遮挡后有效玻璃面积: {glass_area} ㎡")
    print("=" * 60)

    # 保存结果
    result_df = pd.DataFrame({
        '时间': data['hour'][1:],
        '室外温度': outdoor_temp_list,
        '室内温度': indoor_temps[1:],
        '制冷能力(W)': cooling_capacities,
        '输入功率(W)': work_inputs,
        '工作比例': work_ratios,
        '控制信号': control_signals,
        '太阳辐射热(W)': solar_gains
    })
    result_df.to_csv(f'空调系统优化_{target_temp}°C_模拟结果.csv', index=False)

    return result_df


# ================ 普通模式模块 ================
class NormalModeController:
    """定频空调普通模式控制器（修复停机问题）"""
    def __init__(self, target_temp=24.0):
        self.target_temp = target_temp
        self.compressor_on = False  # 压缩机状态
        self.last_ratio = 0.0
        self.startup_energy = 0  # 启动能耗计数器
        self.start_count = 0  # 启动次数计数器
        self.shutdown_tolerance = 0.1  # 停机温度容差

    def calculate_work_ratio(self, indoor_temp, outdoor_temp, elapsed_hours):
        """普通模式工作比例计算（修复停机问题）"""
        # 定频空调只有两种状态: 全功率(1.0)或停止(0.0)
        if indoor_temp > self.target_temp + 1:
            if not self.compressor_on:
                self.start_count += 1  # 记录启动次数
                # 添加启动能耗 (每次启动额外消耗150Wh)
                self.startup_energy += 150
            self.compressor_on = True
            return 1.0
        elif indoor_temp <= (self.target_temp + self.shutdown_tolerance):
            # 修复：当温度达到或低于设定温度+容差时，强制停机
            self.compressor_on = False
            return 0.0  # 压缩机停止时功率直接归零
        else:
            # 保持当前状态
            return 1.0 if self.compressor_on else 0.0


# 普通模式模拟函数
def simulate_normal_mode(target_temp=24.0, env_data=None):  # 增加env_data参数
    # 使用传入的环境数据或生成新数据
    data = env_data if env_data is not None else generate_summer_temperature(start_temp=28.3, duration_hours=24)

    # 创建系统组件
    thermodynamics = ACSystemThermodynamics(max_power=3000)
    efficiency = ACEfficiency()
    # 使用修复后的普通模式控制器
    normal_controller = NormalModeController(target_temp=target_temp)

    # 初始化参数
    # 初始室内温度比室外高2度
    initial_indoor_temp = data.iloc[0]['temperature'] + 2.0
    indoor_temps = [initial_indoor_temp]
    cooling_capacities = []
    outdoor_temp_list = []
    work_inputs = []
    work_ratios = []
    solar_gains = []  # 记录太阳辐射热

    # 热力学参数
    C = 10.0e6  # 房间热容 (J/K)
    K = 35  # 热传导系数 (W/K)
    glass_area = 3.0  # 窗帘遮挡后有效玻璃面积 (平方米)
    glass_transmittance = 0.7  # 玻璃透射率
    total_heat_sources = 1200  # 总热源 = 人类活动400W + 笔记本800W = 1200W
    dt = 900  # 时间步长 (15分钟)

    # 模拟过程
    elapsed_hours = 0.0
    for i in range(1, len(data)):
        current_outdoor_temp = data.iloc[i]['temperature']
        humidity = data.iloc[i]['humidity']
        flow = data.iloc[i]['flow']
        outdoor_temp_list.append(current_outdoor_temp)

        elapsed_hours += 0.25
        current_indoor_temp = indoor_temps[-1]

        # 计算工作比例
        work_ratio = normal_controller.calculate_work_ratio(
            current_indoor_temp,
            current_outdoor_temp,
            elapsed_hours
        )
        work_ratios.append(work_ratio)

        # 质量流量计算
        mass_flow_rate = flow * 0.0012

        # 温度参数
        T_in = current_indoor_temp
        temp_deviation = current_indoor_temp - target_temp
        base_temp_diff = 1.8
        dynamic_adjust = min(1.2, max(0, abs(temp_deviation) * 0.6))
        temp_diff = base_temp_diff + dynamic_adjust
        T_out = T_in - temp_diff

        # 压力参数
        P_in = 2.2
        P_out = 5.8

        # 当工作比例为0时直接返回0功率 ===
        if work_ratio == 0.0:
            # 当工作比例为0时，直接返回0功率
            work_input = 0.0
            cooling_capacity = 0.0
        else:
            # 否则正常计算功率
            work_input = thermodynamics.calculate_power(
                mass_flow_rate, T_in, P_in, T_out, P_out, work_ratio
            )
            cooling_capacity = efficiency.calculate_cooling_capacity(work_input)

        work_inputs.append(work_input)
        cooling_capacities.append(cooling_capacity)

        # 计算太阳辐射热 (动态模型)
        current_hour = elapsed_hours % 24
        # 太阳辐射强度 (W/m²) - 基于时间和室外温度
        if 6 <= current_hour < 18:
            # 正午12点达到峰值
            solar_intensity = 800 * np.sin(np.pi * (current_hour - 6) / 12)
            solar_intensity *= (1 + 0.15 * (current_outdoor_temp - 28))
        else:
            solar_intensity = 0

        solar_gain = solar_intensity * glass_area * glass_transmittance
        solar_gains.append(solar_gain)

        # 计算下一时刻温度
        heat_transfer = K * (current_outdoor_temp - current_indoor_temp)
        dT = (heat_transfer + solar_gain + total_heat_sources - cooling_capacity) * dt / C
        new_indoor_temp = current_indoor_temp + dT
        new_indoor_temp = max(18, min(new_indoor_temp, 35))
        indoor_temps.append(new_indoor_temp)

    # 计算总能耗 (包括启动能耗)
    total_work = sum(work_inputs) * dt / 3600  # Wh
    total_energy = (total_work + normal_controller.startup_energy) / 1000  # kWh

    # 计算平均COP
    valid_power = [w for w in work_inputs if w > 0]
    avg_cop = np.mean([efficiency.calculate_cop(w) for w in valid_power]) if valid_power else 0

    # 温度在目标范围内的比例
    lower_bound = target_temp - 0.5
    upper_bound = target_temp + 0.5
    in_target_range = sum(1 for t in indoor_temps if lower_bound <= t <= upper_bound) / len(indoor_temps) * 100

    # 可视化
    plt.figure(figsize=(16, 10))

    # 温度变化图
    plt.subplot(2, 2, 1)
    plt.plot(data['hour'][1:], indoor_temps[1:], label="室内温度 (°C)", color='blue', linewidth=2)
    plt.plot(data['hour'], data['temperature'], label="室外温度 (°C)", color='red', alpha=0.7)
    plt.axhline(y=target_temp, color='green', linestyle='--', label="目标温度")
    plt.axhspan(target_temp - 0.5, target_temp + 0.5, color='lightgreen', alpha=0.2, label="舒适范围")
    plt.title(f"普通模式: 室内外温度变化 (目标温度: {target_temp}°C)", fontsize=14)
    plt.xlabel("时间 (小时)", fontsize=12)
    plt.ylabel("温度 (°C)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 功率变化图
    plt.subplot(2, 2, 2)
    plt.plot(data['hour'][1:], work_inputs, label="输入功率 (W)", color='purple')
    plt.axhline(y=3000, color='red', linestyle='--', label="最大功率")
    plt.title(f"普通模式: 功率变化 (目标温度: {target_temp}°C)", fontsize=14)
    plt.xlabel("时间 (小时)", fontsize=12)
    plt.ylabel("功率 (W)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 3300)

    # 工作状态图
    plt.subplot(2, 2, 3)
    # 将工作比例转换为压缩机状态 (0=停止, 1=运行)
    compressor_state = [1 if ratio > 0.5 else 0 for ratio in work_ratios]
    plt.step(data['hour'][1:], compressor_state, where='post', label="压缩机状态", color='brown')
    plt.fill_between(data['hour'][1:], compressor_state, 0, color='brown', alpha=0.3, step='post')
    plt.title("普通模式: 压缩机运行状态", fontsize=14)
    plt.xlabel("时间 (小时)", fontsize=12)
    plt.ylabel("状态 (0=停止, 1=运行)", fontsize=12)
    plt.yticks([0, 1], ['停止', '运行'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-0.1, 1.1)

    # 累计能耗图
    plt.subplot(2, 2, 4)
    # 计算累计能耗 (包括启动能耗)
    cumulative_energy = []
    current_energy = 0
    for i, power in enumerate(work_inputs):
        # 每15分钟的能耗 (Wh)
        energy_step = power * dt / 3600
        current_energy += energy_step

        # 如果是本次启动的第一个功率点，添加启动能耗
        if i > 0 and work_ratios[i] == 1 and work_ratios[i - 1] == 0:
            current_energy += 150  # 每次启动额外150Wh

        cumulative_energy.append(current_energy / 1000)  # 转换为kWh

    plt.plot(data['hour'][1:], cumulative_energy, label="累计能耗", color='darkblue')
    plt.title(f"普通模式: 累计能耗 (目标温度: {target_temp}°C)", fontsize=14)
    plt.xlabel("时间 (小时)", fontsize=12)
    plt.ylabel("能耗 (kWh)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'普通模式_{target_temp}°C_综合分析.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印性能数据
    min_indoor_temp = min(indoor_temps) if indoor_temps else 0
    max_indoor_temp = max(indoor_temps) if indoor_temps else 0

    print(f"\n普通模式 (目标温度 {target_temp}°C) 性能分析")
    print("=" * 60)
    print(f"总能耗: {total_energy:.3f} kWh (含启动能耗)")
    print(f"压缩机启动次数: {normal_controller.start_count}次")
    print(f"启动总能耗: {normal_controller.startup_energy / 1000:.3f} kWh")
    print(f"室内温度范围: {min_indoor_temp:.1f}°C 至 {max_indoor_temp:.1f}°C")
    print(f"温度在{lower_bound:.1f}-{upper_bound:.1f}°C范围内的比例: {in_target_range:.1f}%")
    print(f"平均COP: {avg_cop:.2f}")
    print(f"室内热源总量: {total_heat_sources} W (人类活动400W + 笔记本800W)")
    print(f"太阳辐射热(平均): {np.mean(solar_gains):.1f} W")
    print(f"窗帘遮挡后有效玻璃面积: {glass_area} ㎡")
    print("=" * 60)

    # 保存结果
    result_df = pd.DataFrame({
        '时间': data['hour'][1:],
        '室外温度': outdoor_temp_list,
        '室内温度': indoor_temps[1:],
        '输入功率(W)': work_inputs,
        '工作比例': work_ratios,
        '累计能耗(kWh)': cumulative_energy,
        '太阳辐射热(W)': solar_gains
    })
    result_df.to_csv(f'普通模式_{target_temp}°C_模拟结果.csv', index=False)

    return total_energy, avg_cop, in_target_range


# ================ 主函数 ================
if __name__ == "__main__":
    print("空调系统节能模拟程序（优化版）")
    print("=" * 50)

    try:
        target_temp = float(input("请输入目标温度 (°C, 建议范围22-26): ") or "24")
        target_temp = max(20, min(target_temp, 28))  # 限制温度范围
    except:
        target_temp = 24  # 默认温度

    # 模式选择
    print("\n请选择运行模式:")
    print("1. 节能模式")
    print("2. 普通模式")
    print("3. 节能模式与普通模式对比")
    mode = input("请输入选项(1/2/3): ") or "1"

    if mode == "1":
        print(f"\n开始模拟节能模式运行 (目标温度: {target_temp}°C)...")
        simulate_ac_system_with_energy_saving(target_temp=target_temp)
        print("节能模式模拟完成!")

    elif mode == "2":
        print(f"\n开始模拟普通模式运行 (目标温度: {target_temp}°C)...")
        simulate_normal_mode(target_temp=target_temp)
        print("普通模式模拟完成!")

    elif mode == "3":
        # 生成共享环境数据
        shared_env_data = generate_summer_temperature(start_temp=28.3, duration_hours=24)

        print(f"\n开始模拟节能模式运行 (共享环境数据)...")
        # 将环境数据传入节能模式
        result_energy_saving_df = simulate_ac_system_with_energy_saving(
            target_temp=target_temp,
            env_data=shared_env_data
        )

        print(f"\n开始模拟普通模式运行 (共享环境数据)...")
        # 将相同环境数据传入普通模式
        normal_total_energy, normal_avg_cop, normal_in_target_range = simulate_normal_mode(
            target_temp=target_temp,
            env_data=shared_env_data
        )

        # 从节能模式结果中提取实际值
        # 总能耗 = 功率总和 × 时间步长 / 3600000 (转换为kWh)
        energy_saving_total_energy = result_energy_saving_df['输入功率(W)'].sum() * 900 / 3600000

        # 计算平均COP
        power_values = result_energy_saving_df['输入功率(W)']
        efficiency_calc = ACEfficiency()
        cop_values = [efficiency_calc.calculate_cop(power) for power in power_values if power > 0]
        energy_saving_avg_cop = np.mean(cop_values) if cop_values else 0

        # 温度达标率
        indoor_temps = result_energy_saving_df['室内温度']
        lower_bound = target_temp - 0.5
        upper_bound = target_temp + 0.5
        energy_saving_in_target_range = ((indoor_temps >= lower_bound) & (indoor_temps <= upper_bound)).mean() * 100

        # 计算节能率
        saving_rate = (normal_total_energy - energy_saving_total_energy) / normal_total_energy * 100

        # 对比结果
        print("\n两种模式性能对比")
        print("=" * 60)
        print(f"{'指标':<20} | {'节能模式':<10} | {'普通模式':<10} | {'差异':<10}")
        print("-" * 60)
        print(
            f"{'总能耗 (kWh)':<20} | {energy_saving_total_energy:<10.3f} | {normal_total_energy:<10.3f} | -{normal_total_energy - energy_saving_total_energy:.3f} kWh")
        print(
            f"{'平均COP':<20} | {energy_saving_avg_cop:<10.2f} | {normal_avg_cop:<10.2f} | +{energy_saving_avg_cop - normal_avg_cop:.2f}")
        print(
            f"{'温度达标率 (%)':<20} | {energy_saving_in_target_range:<10.1f} | {normal_in_target_range:<10.1f} | +{energy_saving_in_target_range - normal_in_target_range:.1f}%")
        print("=" * 60)
        print(f"节能率: {saving_rate:.1f}%")
        print("=" * 60)

    else:
        print("无效的选择，程序退出")