import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point
from ackermann_msgs.msg import AckermannDriveStamped  # 메시지 타입 확인 필요
from rclpy.qos import qos_profile_sensor_data
import matplotlib.pyplot as plt
import csv
import time
from collections import deque

class SystemValidator(Node):
    def __init__(self):
        super().__init__('system_validator')

        # ==========================================
        # [설정] 본인의 환경에 맞게 수정하세요
        # ==========================================
        self.odom_topic = '/car_state/pose'
        self.drive_topic = '/drive'
        self.control_freq = 40.0  # 제어기 목표 주파수 (Hz)
        self.log_file = 'latency_log.csv'
        # ==========================================

        self.last_odom_stamp = None
        self.last_odom_rcv_time = None
        
        # 데이터 저장용 리스트
        self.time_history = []
        self.data_age_history = []
        self.reception_gap_history = []

        # 1. Odom 구독 (소스 데이터)
        self.create_subscription(
            Odometry, 
            self.odom_topic, 
            self.odom_callback, 
            qos_profile_sensor_data
        )

        # 2. Drive 구독 (실제 제어기 검증용)
        self.create_subscription(
            AckermannDriveStamped, 
            self.drive_topic, 
            self.drive_callback, 
            qos_profile_sensor_data
        )

        # 3. [가설 검증용] 가상 타이머 (Timer 방식 시뮬레이션)
        # 실제 제어기가 꺼져있을 때, 타이머 방식이 얼마나 밀리는지 확인용
        self.timer = self.create_timer(1.0 / self.control_freq, self.mock_timer_loop)
        
        self.start_time = time.time()
        self.get_logger().info(f"검증 시작! 데이터가 {self.log_file}에 저장됩니다...")

    def odom_callback(self, msg):
        # Odom이 생성된 시간 (Header Stamp)
        self.last_odom_stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        # Odom이 도착한 시간 (Reception Time)
        self.last_odom_rcv_time = self.get_logger().get_clock().now()

    def drive_callback(self, msg):
        """실제 제어기가 켜져 있을 때 실행됨"""
        self.record_metrics("Real_Control")

    def mock_timer_loop(self):
        """제어기가 타이머 기반일 때를 시뮬레이션 (Ghost Mode)"""
        # 실제 제어기가 돌고 있다면 이 함수 내용은 무시하거나, 
        # 가설 검증용으로만 별도로 분석할 것.
        # 여기서는 편의상 'Mock' 태그로 기록함.
        self.record_metrics("Mock_Timer")

    def record_metrics(self, source):
        if self.last_odom_stamp is None:
            return

        current_time = self.get_logger().get_clock().now()
        
        # 1. Data Age (데이터 나이): 현재 시각 - Odom 생성 시각
        # "지금 내가 쓰려는 데이터가 얼마나 늙었나?"
        age_ns = (current_time - self.last_odom_stamp).nanoseconds
        age_ms = age_ns / 1e6

        # 2. Reception Gap (수신 시차): 현재 시각 - Odom 도착 시각
        # "데이터가 현관에 도착하고 나서 얼마나 방치되었나?"
        if self.last_odom_rcv_time:
            gap_ns = (current_time - self.last_odom_rcv_time).nanoseconds
            gap_ms = gap_ns / 1e6
        else:
            gap_ms = 0.0

        elapsed_time = time.time() - self.start_time

        # 로그 저장
        self.time_history.append(elapsed_time)
        self.data_age_history.append(age_ms)
        self.reception_gap_history.append(gap_ms)

        # 터미널 출력 (Real Control일 때만, 혹은 필요시)
        if source == "Real_Control":
            print(f"[{source}] Time: {elapsed_time:.2f}s | Age: {age_ms:.1f}ms | Gap: {gap_ms:.1f}ms")

    def save_and_plot(self):
        # 1. CSV 저장
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_Sec', 'Data_Age_ms', 'Reception_Gap_ms'])
            for t, age, gap in zip(self.time_history, self.data_age_history, self.reception_gap_history):
                writer.writerow([t, age, gap])
        
        self.get_logger().info(f"데이터 저장 완료: {self.log_file}")

        # 2. 그래프 그리기
        plt.figure(figsize=(10, 6))
        
        # 데이터 나이 (Data Age)
        plt.subplot(2, 1, 1)
        plt.plot(self.time_history, self.data_age_history, 'r.-', label='Data Age (Latency)')
        plt.ylabel('Latency (ms)')
        plt.title('System Latency Analysis')
        plt.grid(True)
        plt.legend()

        # 수신 시차 (Reception Gap) -> 톱니바퀴 모양 확인용
        plt.subplot(2, 1, 2)
        plt.plot(self.time_history, self.reception_gap_history, 'b.-', label='Reception Gap (Drift)')
        plt.ylabel('Gap (ms)')
        plt.xlabel('Experiment Time (s)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = SystemValidator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("종료 중... 데이터 저장 및 그래프 출력")
        node.save_and_plot()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()