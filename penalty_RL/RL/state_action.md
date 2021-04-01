<!-- Actions -->


<!-- Shoot fast -->
Body_GoToPoint( agent->world().ball().pos(), 0.4, ServerParam::i().maxDashPower()).execute( agent );
Body_TurnToBall().execute( agent );
Body_KickOneStep( shoot_point,ServerParam::i().ballSpeedMax()).execute( agent );


<!-- Shoot -->
Body_SmartKick( shot_point, shot_speed, shot_speed * 0.96, 2 ).execute( agent );
agent->setNeckAction( new Neck_TurnToPoint( shot_point ) ); new Neck_ScanField()


<!-- Dribble -->
Body_TurnToPoint( drib_target ).execute( agent );
Body_StopBall().execute( agent );
Body_Dribble2008( drib_target,2.0,drib_power,drib_dashes,dodge_mode).execute( agent );


<!-- Basic actions -->
agent->doTurn( turn_moment );
agent->doDash( result_dash_powers.front(), dash_dir );
agent->doKick( required_kick_power, required_accel.th() - wm.self().body() );
doTackle?
doTurnNeck?
doMove?
doViewAction?

<!-- State  -->
