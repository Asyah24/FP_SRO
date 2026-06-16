from enum import IntEnum
from dataclasses import dataclass, asdict
import time
import json
import requests
import coppelia_client
import signal

# ==========================================
# 1. CONFIGURATION & MISSION TARGET
# ==========================================
@dataclass
class Config:
    base_speed: float = 0.45         # Kecepatan aman untuk pemrosesan AI
    wall_dist: float = 0.25          # Jarak ideal dari dinding
    front_turn_dist: float = 0.40    # Batas deteksi dinding depan
    emergency_dist: float = 0.15     # Batas darurat terlalu dekat
    
    # Parameter PID (Bugs-Fixed Parameters)
    kp: float = 5.0
    ki: float = 0.05
    kd: float = 3.0
    avoid_gain: float = 4.0
    max_turn: float = 0.7
    integral_limit: float = 0.3
    smoothing_alpha: float = 0.6

    # LLM Interface Configuration
    ollama_url: str = "http://localhost:11434/api/generate"
    model_name: str = "qwen2.5:1.5b"  # Menggunakan model lokal yang ringan
    debug: bool = True

# Definisikan koordinat Misi (Goal Position) sesuai labirinmu
GOAL_POSITION = [2.5, -4.0] 

class Side(IntEnum):
    RIGHT = 1
    LEFT = -1

@dataclass
class SensorData:
    front_left: float; front_center: float; front_right: float
    back_left: float; back_center: float; back_right: float

class SensorProcessor:
    @staticmethod
    def process(readings: list) -> SensorData:
        return SensorData(
            front_left=min(readings[0], readings[1], readings[2]),
            front_center=min(readings[3], readings[4]),
            front_right=min(readings[5], readings[6], readings[7]),
            back_left=min(readings[13], readings[14], readings[15]),
            back_center=min(readings[11], readings[12]),
            back_right=min(readings[8], readings[9], readings[10])
        )

# ==========================================
# 2. CLOSED-LOOP PID CONTROLLER (BUGS FIXED)
# ==========================================
class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.integral_limit = integral_limit
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, error: float, current_time: float) -> float:
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.05  # FIXED: Menggunakan '=' penugasan nilai, bukan ':' type-hint kosong

        p_term = self.kp * error
        
        # FIXED: Akumulasi integral menggunakan '+=' agar menyimpan memori error masa lalu
        self.integral += error * dt 
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        i_term = self.ki * self.integral

        d_term = self.kd * (error - self.prev_error) / dt
 
        self.prev_error = error
        self.last_time = current_time

        return p_term + i_term + d_term

    def reset(self):
        self.prev_error = 0.0; self.integral = 0.0; self.last_time = time.time()

# ==========================================
# 3. AI LLM INTERFACE (STATE -> ACTION ARGUMENTS)
# ==========================================
def ai_agent_interface(state: dict) -> dict:
    """
    Fungsi Interface utama sesuai permintaan Dosen.
    Menerima argumen berupa struktur data 'state' lengkap,
    dan mengembalikan output keputusan 'action' dari LLM.
    """
    config = Config()
    
    # Menyusun prompt berisi seluruh komponen State yang diminta dosen
    prompt = f"""
    [SYSTEM]
    You are the high-level AI brain of a mobile robot PioneerP3DX. 
    Analyze the vehicle state and choose the best macro action.
    Available actions: "MOVE_FORWARD", "FOLLOW_WALL_RIGHT", "TURN_LEFT_90", "REACHED_GOAL".

    [ROBOT STATE]
    - Current Position: X={state['pos_x']:.2f}, Y={state['pos_y']:.2f}
    - Target Goal Position: X={state['goal_x']:.2f}, Y={state['goal_y']:.2f}
    - Distance to Goal: {state['dist_to_goal']:.2f} m
    - Current Speed: {state['speed']:.2f} m/s
    - Sonar Front Center: {state['sensors']['front_center']:.2f} m
    - Sonar Front Right: {state['sensors']['front_right']:.2f} m

    [MISSION RULES]
    1. If Distance to Goal < 0.3, you MUST output "REACHED_GOAL".
    2. If Sonar Front Center < {config.front_turn_dist}, you MUST output "TURN_LEFT_90" to avoid crash.
    3. If Sonar Front Right < 0.6, output "FOLLOW_WALL_RIGHT".
    4. Else, output "MOVE_FORWARD".

    Respond ONLY with JSON format: {{"action": "<chosen_action>", "reason": "<short_reason>"}}
    """
    
    try:
        payload = {"model": config.model_name, "prompt": prompt, "stream": False, "format": "json"}
        response = requests.post(config.ollama_url, json=payload, timeout=1.5)
        ai_decision = json.loads(response.json()['response'].strip())
        return {"action": ai_decision.get("action", "MOVE_FORWARD"), "status": "AI_ACTIVE"}
    except:
        # Fallback System (Kendali Darurat berbasis Aturan Lokal jika LLM timeout/stutter)
        if state['sensors']['front_center'] < config.front_turn_dist:
            return {"action": "TURN_LEFT_90", "status": "FALLBACK_RULE"}
        elif state['sensors']['front_right'] < 0.5:
            return {"action": "FOLLOW_WALL_RIGHT", "status": "FALLBACK_RULE"}
        return {"action": "MOVE_FORWARD", "status": "FALLBACK_RULE"}

# ==========================================
# 4. CORE EXECUTION & BODY-TO-JOINT SPACE
# ==========================================
def main():
    coppelia = coppelia_client.Coppelia()
    robot = coppelia_client.P3DX(coppelia.sim, "PioneerP3DX")
    config = Config()
    pid = PIDController(config.kp, config.ki, config.kd, config.integral_limit)
    
    # Ambil handle object untuk membaca koordinat posisi asli robot di simulator
    robot_handle = coppelia.sim.getObject("/PioneerP3DX")
    
    running = True
    def on_ctrl_c(sig, frame):
        nonlocal running; running = False
    signal.signal(signal.SIGINT, on_ctrl_c)

    print("⏳ Pemanasan VRAM untuk AI LLM Interface...")
    try:
        requests.post(config.ollama_url, json={"model": config.model_name, "prompt": "init", "stream": False}, timeout=5.0)
        print("✅ AI Inference System Ready!")
    except:
        print("⚠️ Server Ollama tidak aktif di background. Menggunakan mode Fallback Dinamis.")

    coppelia.start_simulation()
    loop_counter = 0
    current_action = "MOVE_FORWARD"

    while running and coppelia.is_running():
        now = time.time()
        
        # 1. PENGAMBILAN DATA STATE SEPERTI YANG DIINGINKAN DOSEN
        pos_3d = coppelia.sim.getObjectPosition(robot_handle, -1)
        pos_x, pos_y = pos_3d[0], pos_3d[1]
        dist_to_goal = ((GOAL_POSITION[0] - pos_x)**2 + (GOAL_POSITION[1] - pos_y)**2)**0.5
        sensors = SensorProcessor.process(robot.get_sonar())
        
        # Mengemas seluruh parameter ke dalam objek State tunggal
        robot_state_dataset = {
            "pos_x": pos_x, "pos_y": pos_y,
            "goal_x": GOAL_POSITION[0], "goal_y": GOAL_POSITION[1],
            "dist_to_goal": dist_to_goal,
            "speed": config.base_speed,
            "sensors": asdict(sensors)
        }

        # 2. INFERENCE CALL KE AI INTERFACE (Setiap 6 Loop agar simulasi berjalan smooth)
        if loop_counter % 6 == 0:
            ai_response = ai_agent_interface(robot_state_dataset)
            current_action = ai_response["action"]
            if config.debug:
                print(f"[STATE-ACTION INTERFACE] Mode: {ai_response['status']} | Action: {current_action} | Dist to Goal: {dist_to_goal:.2f}m")

        # 3. BODY SPACE TRANSLATION TO JOINT SPACE SPEED
        left_speed, right_speed = config.base_speed, config.base_speed
        
        if current_action == "REACHED_GOAL":
            left_speed, right_speed = 0.0, 0.0
            print("🏁 MISI SUKSES: Robot Telah Mencapai Posisi Koordinat Target Goal!")
            running = False
            
        elif current_action == "TURN_LEFT_90":
            # Berputar tajam menjauhi dinding depan
            left_speed, right_speed = -config.base_speed * 0.5, config.base_speed * 0.5
            
        elif current_action == "FOLLOW_WALL_RIGHT":
            # Eksekusi kendali closed-loop PID untuk menyusuri dinding kanan
            dist_error = sensors.front_right - config.wall_dist
            turn = pid.compute(dist_error, now) * -1.0 # Side Right = 1
            
            # Tambah proteksi sisi kiri jika terlalu mepet
            avoid_error = max(0.0, (config.wall_dist * 1.2) - sensors.front_left)
            avoid_turn = avoid_error * config.avoid_gain * -1.0
            
            total_turn = turn + avoid_turn
            max_turn_effect = config.max_turn * config.base_speed
            total_turn = max(-max_turn_effect, min(max_turn_effect, total_turn))
            
            left_speed = config.base_speed - total_turn
            right_speed = config.base_speed + total_turn

        # 4. KIRIM ACTION KE AKT_UATOR COPPELIA
        robot.set_speed(left_speed, right_speed)
        
        loop_counter += 1
        time.sleep(0.04)

    robot.set_speed(0, 0)
    coppelia.stop_simulation()
    print("Simulasi Dihentikan Selesai.")

if __name__ == "__main__":
    main()