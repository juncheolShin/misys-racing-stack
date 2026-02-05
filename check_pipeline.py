import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped # 혹은 사용하시는 Drive 메시지 타입
from rclpy.qos import qos_profile_sensor_data

class PipelineValidator(Node):
    def __init__(self):
        super().__init__('pipeline_validator')
        
        # 1. 토픽 이름 설정 (사용하시는 환경에 맞게 수정하세요)
        self.odom_topic = '/car_state/pose'             # State Estimation 결과
        self.drive_topic = '/drive'                     # 제어기 출력
        
        # 2. 마지막으로 수신한 오도메트리 시간 저장용
        self.last_odom_receive_time = None
        self.last_odom_header_stamp = None
        
        # 3. 구독자 설정
        self.create_subscription(
            PoseStamped, 
            self.odom_topic, 
            self.odom_callback, 
            qos_profile_sensor_data
        )
        
        self.create_subscription(
            AckermannDriveStamped, 
            self.drive_topic, 
            self.drive_callback, 
            qos_profile_sensor_data
        )

        self.get_logger().info("검증기가 시작되었습니다. 메시지 순서를 감시합니다...")

    def odom_callback(self, msg):
        # State Estimation이 데이터를 쏘는 순간 기록
        # 1. 실제 메시지에 찍힌 시간 (Source Time)
        self.last_odom_header_stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        # 2. 내 노드가 받은 시간 (Arrival Time)
        self.last_odom_receive_time = self.get_clock().now()

    def drive_callback(self, msg):
        current_time = self.get_clock().now()
        drive_stamp = rclpy.time.Time.from_msg(msg.header.stamp)

        # 오도메트리가 아직 한 번도 안 왔으면 패스
        if self.last_odom_header_stamp is None:
            return

        # --- 분석 시작 ---
        
        # 1. 데이터 나이 (Data Age): 제어 명령이 나갈 때, 위치 데이터가 얼마나 늙었나?
        # (제어 명령 발행 시각 - 위치 데이터 생성 시각)
        # 이상적 값: 연산 시간 (약 15~20ms)
        data_age_ns = (drive_stamp - self.last_odom_header_stamp).nanoseconds
        data_age_ms = data_age_ns / 1e6
        
        # 2. 수신 순서 체크 (Reception Order)
        # 제어 명령을 받았을 때, 방금 받은 오도메트리가 그 직전 것인가?
        # (수신 시각 차이)
        reception_gap_ns = (current_time - self.last_odom_receive_time).nanoseconds
        reception_gap_ms = reception_gap_ns / 1e6
        
        # --- 판정 로직 ---
        status = "✅ PASS"
        
        # 기준 1: 데이터가 너무 늙었음 (40Hz 기준 25ms가 주기인데 30ms 이상이면 위험)
        if data_age_ms > 30.0: 
            status = "⚠️ STALE (지연 발생)"
        
        # 기준 2: 순서 역전 (이론상 불가능하지만, 제어기가 이전 틱의 데이터를 썼다면 발생 가능)
        if data_age_ms < 0:
            status = "❌ FAIL (미래의 데이터? 시간 동기화 문제)"

        # 기준 3: 한참 뒤에 명령이 옴 (State Est는 빨리 왔는데 제어기가 너무 늦게 반응)
        if reception_gap_ms > 50.0:
            status = "⚠️ LAG (연산 과부하 의심)"

        # --- 로그 출력 ---
        self.get_logger().info(
            f"[{status}] "
            f"Odom->Cmd 간격(Data Age): {data_age_ms:.2f}ms | "
            f"수신 시차: {reception_gap_ms:.2f}ms"
        )

def main(args=None):
    rclpy.init(args=args)
    node = PipelineValidator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()