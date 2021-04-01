# thrift -out /robocup/agent2DRL/src/RL -r --gen cpp RL.thrift

struct State {
    1: required i32 cycle;
    2: string message;

    3: bool is_kickable; //wm.self().isKickable()
    4: double ball_pos_x; //wm.ball().pos()
    5: double ball_pos_y; //wm.ball().pos()
    6: double ball_vel_x; // wm.ball().vel()
    7: double ball_vel_y;  // wm.ball().vel()
    8: i32 ball_pos_count;// wm.ball().posCount()

    // const InterceptTable * table = wm.interceptTable(); table->selfReachCycle() > 100
    // Vector2D final_point = wm.ball().inertiaFinalPoint();
    9: double self_stamina;// wm.self().stamina()

    // const AbstractPlayerObject * opp_goalie = agent->world().getTheirGoalie(); opp_goalie->body().degree() opp_goalie->pos().y
    10: double goalie_pos_x;// opp->pos()
    11: double goalie_pos_y;
    12: double goalie_vel_x;// opp->vel()
    13: double goalie_vel_y;
    14: double goalie_body;


    15: double self_pos_x;// wm.self().pos().x
    16: double self_pos_y;// wm.self().pos().y
    17: double self_vel_x;// wm.self().vel().x
    18: double self_vel_y;// wm.self().vel().y
    19: double self_body;//  wm.self().body()
    20: double self_dash_rate;// wm.self().dashRate()

    21: i32 time_spent;// wm.time().cycle() - state->time().cycle() //Spent time to compare to ServerParam::i().penTakenWait()

    22: i32 self_unum;

}
// Interesting fixed constants:
// ServerParam::i().maxDashPower()
// wm.self().playerType().playerDecay()
// wm.self().playerType().playerSpeedMax()
// ServerParam::i().penTakenWait()

struct Action {
  1: i32 cycle;
  2: string name;
  3: double dash_power;
  4: double dash_dir;
  5: double kick_power;
  6: double kick_direction;
  7: double move_x;
  8: double move_y;
  9: double turn_moment;
  10: double turn_neck_angle;
  11: i32 unum;
}

service RL {
    void send_state(1:State state),
    void send_action(1:Action action),
    void send_start_signal(1:i32 cycle, 2:string gamemode, 3:i32 side, 4:i32 unum),
    void send_terminal_signal(1:i32 cycle, 2:string gamemode, 3:i32 side),
    Action act(1:State state),
}
