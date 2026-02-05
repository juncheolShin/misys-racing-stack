#ifndef DELAY_COMPENSATOR_HPP
#define DELAY_COMPENSATOR_HPP

#include <deque>
#include <cmath>
#include <tuple>
#include <algorithm> 
#include <rclcpp/rclcpp.hpp>

namespace controller_utils {

struct CmdHistory {
    double steering;
    double velocity;
    double dt;
};

class DelayCompensator {
public:
    /**
     * @brief 생성자
     * @param delay_time  보상할 지연 시간 
     * @param wheelbase   차량 축거
     * @param logger     
     */
    DelayCompensator(double delay_time, double wheelbase, rclcpp::Logger logger) 
        : delay_time_(delay_time), wheelbase_(wheelbase), logger_(logger) {}

    /**
     * @brief 지연 시간 설정 변경
     */
    void set_delay(double delay_time) {
        delay_time_ = delay_time;
    }

    /**
     * @brief
     * @param x, y, yaw  현재 차량 위치
     * @param v          현재 차량 속도
     * @param dt         제어 주기 
     * @return std::tuple<x, y, yaw, v> 
     */
    std::tuple<double, double, double, double> compensate(
        double x, double y, double yaw, double v, double dt) {
        
        double pred_x = x;
        double pred_y = y;
        double pred_yaw = yaw;
        double pred_v = v;

        // 지연 보상 로직
        if (dt > 1e-6 && !cmd_queue_.empty()) {
            
            // 지연 시간만큼 거슬러 올라가기 위한 스텝 수 계산
            // (가정: 큐에 쌓인 명령들은 dt 간격으로 실행될 예정임)
            int steps = (int)(delay_time_ / dt);

            // 큐 크기를 초과하지 않도록 안전 장치
            int queue_size = static_cast<int>(cmd_queue_.size());
            int start_idx = std::max(0, queue_size - steps);

            for (int i = start_idx; i < queue_size; ++i) {
                const auto& cmd = cmd_queue_[i];

                // --- Kinematic Bicycle Model ---
                
                double beta = std::atan(0.5 * std::tan(cmd.steering));
                pred_x += cmd.velocity * std::cos(pred_yaw + beta) * cmd.dt;
                pred_y += cmd.velocity * std::sin(pred_yaw + beta) * cmd.dt;
                pred_yaw += (cmd.velocity / wheelbase_) * std::sin(beta) * 2.0 * cmd.dt;
                pred_v = cmd.velocity; 
            }
        }

        double diff_dist = std::sqrt(std::pow(pred_x - x, 2) + std::pow(pred_y - y, 2));
        double diff_yaw = pred_yaw - yaw;
        static rclcpp::Clock loop_clock(RCL_STEADY_TIME);

        RCLCPP_INFO_THROTTLE(logger_, loop_clock, 1000, 
            "[DelayComp] Delay: %.3fs | Comp_Dist: %.3fm | Comp_Yaw: %.3frad", 
            delay_time_, diff_dist, diff_yaw);

        return {pred_x, pred_y, pred_yaw, pred_v};
    }

    /**
     * @brief 제어 루프 마지막에 호출하여 명령 큐 업데이트
     */
    void update_queue(double steering, double velocity, double dt) {
        // 새 명령 추가
        cmd_queue_.push_back({steering, velocity, dt});

        // 큐 관리
        double max_history_time = 1.0; 
        size_t max_size = (size_t)(max_history_time / std::max(dt, 0.001));
        
        while (cmd_queue_.size() > max_size) {
            cmd_queue_.pop_front();
        }
    }

private:
    std::deque<CmdHistory> cmd_queue_;
    double delay_time_;
    double wheelbase_;
    rclcpp::Logger logger_;
};

} // namespace controller_utils

#endif // DELAY_COMPENSATOR_HPP