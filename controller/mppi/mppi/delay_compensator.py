"""
Delay Compensator for MPPI Controller

Ported from: controller_manager/include/latency_compensator.hpp
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Tuple
import rclpy


@dataclass
class CmdHistory:
    """과거 명령 히스토리 저장"""
    steering: float
    velocity: float
    dt: float


class DelayCompensator:
    """
    지연 보상 클래스
    
    과거 명령 히스토리를 사용하여 현재 차량 상태를
    지연 시간만큼 예측하여 보상합니다.
    """
    
    def __init__(self, delay_time: float, wheelbase: float, logger=None):
        """
        Args:
            delay_time: 보상할 지연 시간 [s]
            wheelbase: 차량 축거 [m]
            logger: ROS2 logger (optional)
        """
        self.delay_time = delay_time
        self.wheelbase = wheelbase
        self.logger = logger
        self.cmd_queue: deque = deque()
        
    def set_delay(self, delay_time: float):
        """지연 시간 설정 변경"""
        self.delay_time = delay_time
        
    def compensate(self, x: float, y: float, yaw: float, v: float, dt: float) -> Tuple[float, float, float, float]:
        """
        지연 보상 수행
        
        Args:
            x, y, yaw: 현재 차량 위치
            v: 현재 차량 속도
            dt: 제어 주기
            
        Returns:
            (pred_x, pred_y, pred_yaw, pred_v): 예측된 상태
        """
        pred_x = x
        pred_y = y
        pred_yaw = yaw
        pred_v = v
        
        # 지연 보상 로직
        if dt > 1e-6 and len(self.cmd_queue) > 0:
            # 지연 시간만큼 거슬러 올라가기 위한 스텝 수 계산
            # (가정: 큐에 쌓인 명령들은 dt 간격으로 실행될 예정임)
            steps = int(self.delay_time / dt)
            
            # 큐 크기를 초과하지 않도록 안전 장치
            queue_size = len(self.cmd_queue)
            start_idx = max(0, queue_size - steps)
            
            # 큐를 리스트로 변환하여 인덱싱
            queue_list = list(self.cmd_queue)
            
            for i in range(start_idx, queue_size):
                cmd = queue_list[i]
                
                # --- Kinematic Bicycle Model 적분 ---
                beta = math.atan(0.5 * math.tan(cmd.steering))
                pred_x += cmd.velocity * math.cos(pred_yaw + beta) * cmd.dt
                pred_y += cmd.velocity * math.sin(pred_yaw + beta) * cmd.dt
                pred_yaw += (cmd.velocity / self.wheelbase) * math.sin(beta) * 2.0 * cmd.dt
                pred_v = cmd.velocity
        
        # 디버그 로깅
        diff_dist = math.sqrt((pred_x - x)**2 + (pred_y - y)**2)
        diff_yaw = pred_yaw - yaw
        
        if self.logger is not None:
            self.logger.info(
                f"[DelayComp] Delay: {self.delay_time:.3f}s | Comp_Dist: {diff_dist:.3f}m | Comp_Yaw: {diff_yaw:.3f}rad",
                throttle_duration_sec=1.0
            )
        
        return (pred_x, pred_y, pred_yaw, pred_v)
    
    def update_queue(self, steering: float, velocity: float, dt: float):
        """
        제어 루프 마지막에 호출하여 명령 큐 업데이트
        
        Args:
            steering: 조향각 명령
            velocity: 속도 명령
            dt: 제어 주기
        """
        # 새 명령 추가
        self.cmd_queue.append(CmdHistory(steering=steering, velocity=velocity, dt=dt))
        
        # 큐 관리
        max_history_time = 1.0
        max_size = int(max_history_time / max(dt, 0.001))
        
        while len(self.cmd_queue) > max_size:
            self.cmd_queue.popleft()
