service PingPong {
    string ping(1:string msg),
    double evaluate(1:double msg),
}

struct PlayerState {
  1: required double x;
  2: required double y;
  3: required i32 unum;
}

struct FieldEvaluation {
    1: required i32 cycle;
    2: required double res;
    3: required i32 holder_unum;
    4: required i32 wm_self_unum;
    5: required i32 state_self_unum;
    6: required double wm_ball_x;
    7: required double wm_ball_y;
    8: required double state_ball_x;
    9: required double state_ball_y;
    10: required double theirTeamGoalPos_x;
    11: required double theirTeamGoalPos_y;
    12: required double wm_self_x;
    13: required double wm_self_y;
    14: required double holder_x;
    15: required double holder_y;
    16: required list<PlayerState> opps;
    17: required list<PlayerState> mates;
    18: required i32 action_type;
}

typedef list<FieldEvaluation> FieldEvaluationList

service RoboCup {
    void save_field_evaluations(1:FieldEvaluationList list_),
}
