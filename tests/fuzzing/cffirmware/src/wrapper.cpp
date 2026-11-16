#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

extern "C" {
#include "controller_mellinger.h"
#include "physicalConstants.h"
#include "platform_defaults.h"
#include "power_distribution.h"
#include "stabilizer_types.h"
}

namespace py = pybind11;

namespace {
constexpr float kRadToDeg = 57.295779513082320876f;
constexpr float kQuaternionNormTolerance = 1.0e-3f;
constexpr float kLegacyThrustPwmMin = -2147418112.0f;
constexpr float kLegacyThrustPwmMax = 2147417984.0f;

static_assert(std::numeric_limits<float>::is_iec559);
static_assert(std::numeric_limits<float>::digits == 24);

void require_size(
    const std::vector<float>& values, std::size_t expected, const char* name) {
    if (values.size() != expected) {
        throw py::value_error(
            std::string(name) + " must contain exactly " +
            std::to_string(expected) + " values");
    }
}

void require_finite(const std::vector<float>& values, const char* name) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            throw py::value_error(
                std::string(name) + "[" + std::to_string(index) +
                "] must be finite");
        }
    }
}

float require_legacy_thrust_pwm(const py::handle& value) {
    if (PyBool_Check(value.ptr()) ||
        (!PyFloat_Check(value.ptr()) && !PyLong_Check(value.ptr()))) {
        throw py::type_error("thrust_pwm must be a real Python int or float");
    }
    const double converted = PyFloat_AsDouble(value.ptr());
    if (converted == -1.0 && PyErr_Occurred()) {
        throw py::error_already_set();
    }
    if (!std::isfinite(converted) ||
        converted < -static_cast<double>(FLT_MAX) ||
        converted > static_cast<double>(FLT_MAX)) {
        throw py::value_error("thrust_pwm must be a finite firmware float");
    }
    const float firmware_value = static_cast<float>(converted);
    if (firmware_value < kLegacyThrustPwmMin ||
        firmware_value > kLegacyThrustPwmMax) {
        throw py::value_error(
            "thrust_pwm is outside the safe raw int32 mixer range");
    }
    return firmware_value;
}

std::int16_t require_legacy_int16(const py::handle& value, const char* name) {
    if (PyBool_Check(value.ptr()) || !PyLong_Check(value.ptr())) {
        throw py::type_error(std::string(name) + " must be a Python int");
    }
    const long long converted = PyLong_AsLongLong(value.ptr());
    if (converted == -1 && PyErr_Occurred()) {
        throw py::error_already_set();
    }
    if (converted < std::numeric_limits<std::int16_t>::min() ||
        converted > std::numeric_limits<std::int16_t>::max()) {
        throw py::value_error(std::string(name) + " is outside int16");
    }
    return static_cast<std::int16_t>(converted);
}

py::tuple mix_legacy_raw(
    const py::object& thrust_pwm,
    const py::object& roll_pwm,
    const py::object& pitch_pwm,
    const py::object& yaw_pwm) {
    control_t control{};
    control.controlMode = controlModeLegacy;
    control.thrust = require_legacy_thrust_pwm(thrust_pwm);
    control.roll = require_legacy_int16(roll_pwm, "roll_pwm");
    control.pitch = require_legacy_int16(pitch_pwm, "pitch_pwm");
    control.yaw = require_legacy_int16(yaw_pwm, "yaw_pwm");

    motors_thrust_uncapped_t raw{};
    powerDistribution(&control, &raw);
    return py::make_tuple(
        static_cast<std::int32_t>(raw.motors.m1),
        static_cast<std::int32_t>(raw.motors.m2),
        static_cast<std::int32_t>(raw.motors.m3),
        static_cast<std::int32_t>(raw.motors.m4));
}
}  // namespace

class MellingerFuzzOracle {
 public:
    MellingerFuzzOracle() { controllerMellingerInit(&controller_state_); }

    void reset() { controllerMellingerInit(&controller_state_); }

    py::dict get_params() const {
        py::dict result;
        result["mass"] = controller_state_.mass;
        result["mass_thrust"] = controller_state_.massThrust;
        result["kp_z"] = controller_state_.kp_z;
        result["kd_z"] = controller_state_.kd_z;
        result["ki_z"] = controller_state_.ki_z;
        result["i_range_z"] = controller_state_.i_range_z;
        result["kp_xy"] = controller_state_.kp_xy;
        result["kd_xy"] = controller_state_.kd_xy;
        result["ki_xy"] = controller_state_.ki_xy;
        result["i_range_xy"] = controller_state_.i_range_xy;
        result["kR_xy"] = controller_state_.kR_xy;
        result["kw_xy"] = controller_state_.kw_xy;
        result["ki_m_xy"] = controller_state_.ki_m_xy;
        result["i_range_m_xy"] = controller_state_.i_range_m_xy;
        result["kR_z"] = controller_state_.kR_z;
        result["kw_z"] = controller_state_.kw_z;
        result["ki_m_z"] = controller_state_.ki_m_z;
        result["i_range_m_z"] = controller_state_.i_range_m_z;
        result["kd_omega_rp"] = controller_state_.kd_omega_rp;
        return result;
    }

    void set_internal_state(
        float i_error_x,
        float i_error_y,
        float i_error_z,
        float i_error_m_x,
        float i_error_m_y,
        float i_error_m_z,
        float prev_omega_roll,
        float prev_omega_pitch,
        float prev_setpoint_omega_roll,
        float prev_setpoint_omega_pitch) {
        const std::vector<float> values = {
            i_error_x,
            i_error_y,
            i_error_z,
            i_error_m_x,
            i_error_m_y,
            i_error_m_z,
            prev_omega_roll,
            prev_omega_pitch,
            prev_setpoint_omega_roll,
            prev_setpoint_omega_pitch,
        };
        require_finite(values, "internal_state");
        controller_state_.i_error_x = i_error_x;
        controller_state_.i_error_y = i_error_y;
        controller_state_.i_error_z = i_error_z;
        controller_state_.i_error_m_x = i_error_m_x;
        controller_state_.i_error_m_y = i_error_m_y;
        controller_state_.i_error_m_z = i_error_m_z;
        controller_state_.prev_omega_roll = prev_omega_roll;
        controller_state_.prev_omega_pitch = prev_omega_pitch;
        controller_state_.prev_setpoint_omega_roll =
            prev_setpoint_omega_roll;
        controller_state_.prev_setpoint_omega_pitch =
            prev_setpoint_omega_pitch;
    }

    py::dict get_internal_state() const {
        py::dict result;
        result["i_error_x"] = controller_state_.i_error_x;
        result["i_error_y"] = controller_state_.i_error_y;
        result["i_error_z"] = controller_state_.i_error_z;
        result["i_error_m_x"] = controller_state_.i_error_m_x;
        result["i_error_m_y"] = controller_state_.i_error_m_y;
        result["i_error_m_z"] = controller_state_.i_error_m_z;
        result["prev_omega_roll"] = controller_state_.prev_omega_roll;
        result["prev_omega_pitch"] = controller_state_.prev_omega_pitch;
        return result;
    }

    py::tuple compute_control(
        const std::vector<float>& state,
        const std::vector<float>& setpoint) {
        require_size(state, 13, "state");
        require_size(setpoint, 13, "setpoint");
        require_finite(state, "state");
        require_finite(setpoint, "setpoint");

        const float quaternion_norm = std::sqrt(
            state[6] * state[6] + state[7] * state[7] +
            state[8] * state[8] + state[9] * state[9]);
        if (std::abs(quaternion_norm - 1.0f) >
            kQuaternionNormTolerance) {
            throw py::value_error(
                "state quaternion must be normalized within 1e-3");
        }

        state_t firmware_state{};
        firmware_state.position.x = state[0];
        firmware_state.position.y = state[1];
        firmware_state.position.z = state[2];
        firmware_state.velocity.x = state[3];
        firmware_state.velocity.y = state[4];
        firmware_state.velocity.z = state[5];
        firmware_state.attitudeQuaternion.x = state[6];
        firmware_state.attitudeQuaternion.y = state[7];
        firmware_state.attitudeQuaternion.z = state[8];
        firmware_state.attitudeQuaternion.w = state[9];

        sensorData_t sensors{};
        sensors.gyro.x = state[10] * kRadToDeg;
        sensors.gyro.y = state[11] * kRadToDeg;
        sensors.gyro.z = state[12] * kRadToDeg;

        setpoint_t firmware_setpoint{};
        firmware_setpoint.position.x = setpoint[0];
        firmware_setpoint.position.y = setpoint[1];
        firmware_setpoint.position.z = setpoint[2];
        firmware_setpoint.velocity.x = setpoint[3];
        firmware_setpoint.velocity.y = setpoint[4];
        firmware_setpoint.velocity.z = setpoint[5];
        firmware_setpoint.acceleration.x = setpoint[6];
        firmware_setpoint.acceleration.y = setpoint[7];
        firmware_setpoint.acceleration.z = setpoint[8];
        firmware_setpoint.attitude.yaw = setpoint[9] * kRadToDeg;
        firmware_setpoint.attitudeRate.roll = setpoint[10] * kRadToDeg;
        firmware_setpoint.attitudeRate.pitch = setpoint[11] * kRadToDeg;
        firmware_setpoint.attitudeRate.yaw = setpoint[12] * kRadToDeg;
        firmware_setpoint.mode.x = modeAbs;
        firmware_setpoint.mode.y = modeAbs;
        firmware_setpoint.mode.z = modeAbs;
        firmware_setpoint.mode.yaw = modeAbs;
        firmware_setpoint.mode.quat = modeDisable;
        firmware_setpoint.mode.roll = modeDisable;
        firmware_setpoint.mode.pitch = modeDisable;

        control_t output{};
        controllerMellinger(
            &controller_state_,
            &output,
            &firmware_setpoint,
            &sensors,
            &firmware_state,
            static_cast<stabilizerStep_t>(0));
        return py::make_tuple(
            output.thrust,
            static_cast<int>(output.roll),
            static_cast<int>(output.pitch),
            static_cast<int>(output.yaw));
    }

 private:
    controllerMellinger_t controller_state_{};
};

PYBIND11_MODULE(mellinger_fuzz_oracle, module) {
    module.doc() =
        "Transient pinned Mellinger controller and raw mixer fuzz oracle";
    module.attr("attitude_rate_hz") = ATTITUDE_RATE;
    module.attr("platform_mass_kg") =
        static_cast<float>(CF_MASS);
    module.attr("gravity_m_s2") = static_cast<float>(GRAVITY_MAGNITUDE);
    module.attr("mixer_boundary") =
        "raw uncapped M1-M4 after the pinned Legacy-X matrix; pre_battery, "
        "pre_powerDistributionCap, not final motor PWM";
    module.attr("mixer_motor_order") = "M1,M2,M3,M4";
    module.attr("mixer_raw") = true;
    module.attr("mixer_uncapped") = true;
    module.attr("mixer_pre_battery") = true;
    module.attr("mixer_final_motor_pwm") = false;
    module.attr("mixer_thrust_pwm_float32_safe_range") =
        py::make_tuple(kLegacyThrustPwmMin, kLegacyThrustPwmMax);

    module.def(
        "mix_legacy_raw",
        &mix_legacy_raw,
        py::arg("thrust_pwm"),
        py::arg("roll_pwm"),
        py::arg("pitch_pwm"),
        py::arg("yaw_pwm"));

    py::class_<MellingerFuzzOracle>(module, "MellingerController")
        .def(py::init<>())
        .def("reset", &MellingerFuzzOracle::reset)
        .def("get_params", &MellingerFuzzOracle::get_params)
        .def(
            "set_internal_state",
            &MellingerFuzzOracle::set_internal_state,
            py::arg("i_error_x"),
            py::arg("i_error_y"),
            py::arg("i_error_z"),
            py::arg("i_error_m_x"),
            py::arg("i_error_m_y"),
            py::arg("i_error_m_z"),
            py::arg("prev_omega_roll"),
            py::arg("prev_omega_pitch"),
            py::arg("prev_setpoint_omega_roll"),
            py::arg("prev_setpoint_omega_pitch"))
        .def(
            "get_internal_state",
            &MellingerFuzzOracle::get_internal_state)
        .def(
            "compute_control",
            &MellingerFuzzOracle::compute_control,
            py::arg("state"),
            py::arg("setpoint"));
}
