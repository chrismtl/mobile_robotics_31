import tdmclient.notebook


def global_navigation():
    # Placeholder logic
    print("Global navigation enabled")
    leds_top = [0,0,0]
    motor_left_target = speed0
    motor_right_target = speed0
    # Placeholder logic
    print("Global navigation executed")


def collision_avoidance():
    print(state)
    leds_top = [30,30,30]
    w_l = [40,  20, -20, -20, -40,  30, -10]
    w_r = [-40, -20, -20,  20,  40, -10,  30]
    actuation = [0,0]
    x = [0,0,0,0,0,0,0]

    sensor_translation = 200

    for i in range (6):
        
        x[i] = prox[i] // sensor_translation

        actuation[0] = actuation[0] + x[i] * w_l[i]
        actuation[1] = actuation[1] + x[i] * w_r[i]

    motor_left_target = speed0 + actuation[0]
    motor_right_target = speed0 + actuation[1]

        