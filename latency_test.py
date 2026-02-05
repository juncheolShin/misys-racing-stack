import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import time
import numpy as np
import csv
import os

class RealisticLatencyTester(Node):
    def __init__(self):
        super().__init__('realistic_latency_tester')
        
        # 1. QoS 설정
        qos_depth = 1
        
        # 2. 토픽 설정
        self.pub = self.create_publisher(Odometry, '/dummy/odom', qos_depth)
        self.sub = self.create_subscription(Odometry, '/dummy/odom', self.listener_callback, qos_depth)
        
        # 3. 주기: 40Hz
        self.timer = self.create_timer(1.0/40.0, self.timer_callback)
        
        # 통계 및 저장용 변수
        self.latency_buffer = []  # 실시간 통계 계산용 (100개씩 비움)
        self.all_data = []        # CSV 저장용 전체 데이터 (계속 쌓음)
        self.start_time = time.time()
        
        self.filename = 'latency_result.csv'
        self.get_logger().info(f"측정 시작! 종료 시 {self.filename}에 저장됩니다.")

    def timer_callback(self):
        msg = Odometry()
        
        # [핵심 1] 발행 직전 시간 (순수 통신 지연)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        
        # [핵심 2] 데이터 꽉 채우기 (직렬화 부하)
        dummy_covariance = list(np.random.rand(36))
        msg.pose.covariance = dummy_covariance
        msg.twist.covariance = dummy_covariance
        
        msg.pose.pose.position.x = 1.23456
        msg.pose.pose.orientation.w = 1.0
        msg.twist.twist.linear.x = 5.5
        
        self.pub.publish(msg)

    def listener_callback(self, msg):
        now = self.get_clock().now()
        sent_time = rclpy.time.Time.from_msg(msg.header.stamp)
        
        # 나노초 -> 밀리초 변환
        latency_ms = (now - sent_time).nanoseconds / 1e6
        
        # 경과 시간 (X축 용)
        elapsed_time = time.time() - self.start_time
        
        # 1. CSV 저장용 데이터 확보 (RAM에 저장)
        self.all_data.append([elapsed_time, latency_ms])
        
        # 2. 터미널 출력용 버퍼
        self.latency_buffer.append(latency_ms)
        
        # 100개마다 로그 출력
        if len(self.latency_buffer) >= 100:
            avg_latency = sum(self.latency_buffer) / len(self.latency_buffer)
            max_latency = max(self.latency_buffer)
            self.get_logger().info(
                f"📈 [t={elapsed_time:.1f}s] 평균: {avg_latency:.3f}ms | 최대(Jitter): {max_latency:.3f}ms"
            )
            self.latency_buffer = [] # 버퍼 초기화

    def save_to_csv(self):
        """종료 시 호출되어 데이터를 파일로 씀"""
        if not self.all_data:
            self.get_logger().warn("저장할 데이터가 없습니다.")
            return

        self.get_logger().info(f"데이터 저장 중... ({len(self.all_data)}개 샘플)")
        
        try:
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # 헤더 작성
                writer.writerow(['Time_Sec', 'Latency_ms'])
                # 데이터 작성
                writer.writerows(self.all_data)
            
            self.get_logger().info(f"✅ 저장 완료: {os.path.abspath(self.filename)}")
        except Exception as e:
            self.get_logger().error(f"저장 실패: {e}")

def main():
    rclpy.init()
    np.random.seed(int(time.time()))
    
    node = RealisticLatencyTester()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("종료 요청 받음.")
    finally:
        # 종료 시 CSV 저장 함수 호출
        node.save_to_csv()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()