import gin.tf


''' GYM CLASSIC CONTROL '''
gin.constant('environment_metadata.MOUNTAINCARCONTINUOUS_OBSERVATION_SHAPE', (2, 1))
gin.constant('environment_metadata.MOUNTAINCARCONTINUOUS_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.MOUNTAINCARCONTINUOUS_TIMELIMIT', 999)
gin.constant('environment_metadata.MOUNTAINCARCONTINUOUS_ENV_VER', 'v0')


''' GYM BOX2D OBSERVATION SPACE '''
gin.constant('environment_metadata.BIPEDALWALKER_OBSERVATION_SHAPE', (24, 1))
gin.constant('environment_metadata.BIPEDALWALKERHARDCORE_OBSERVATION_SHAPE', (24, 1))
gin.constant('environment_metadata.LUNARLANDERCONTINUOUS_OBSERVATION_SHAPE', (8, 1))

''' GYM BOX2D ACTION SPACE '''
gin.constant('environment_metadata.BIPEDALWALKER_ACTION_SHAPE', (4,))
gin.constant('environment_metadata.BIPEDALWALKERHARDCORE_ACTION_SHAPE', (4,))
gin.constant('environment_metadata.LUNARLANDERCONTINUOUS_ACTION_SHAPE', (2,))

''' GYM BOX2D TIME LIMIT '''
gin.constant('environment_metadata.BIPEDALWALKER_TIMELIMIT', 1600)
gin.constant('environment_metadata.BIPEDALWALKERHARDCORE_TIMELIMIT', 2000)
gin.constant('environment_metadata.LUNARLANDERCONTINUOUS_TIMELIMIT', 1000)

''' GYM BOX2D ENVIRONMENT VERSION '''
gin.constant('environment_metadata.BIPEDALWALKER_ENV_VER', 'v3')
gin.constant('environment_metadata.BIPEDALWALKERHARDCORE_ENV_VER', 'v3')
gin.constant('environment_metadata.LUNARLANDERCONTINUOUS_ENV_VER', 'v2')


''' GYM MUJOCO OBSERVATION SPACE '''
gin.constant('environment_metadata.ANT_OBSERVATION_SHAPE', (111, 1))
gin.constant('environment_metadata.HALFCHEETAH_OBSERVATION_SHAPE', (17, 1))
gin.constant('environment_metadata.HOPPER_OBSERVATION_SHAPE', (11, 1))
gin.constant('environment_metadata.HUMANOID_OBSERVATION_SHAPE', (376, 1))
gin.constant('environment_metadata.HUMANOIDSTANDUP_OBSERVATION_SHAPE', (376, 1))
gin.constant('environment_metadata.INVERTEDDOUBLEPENDULUM_OBSERVATION_SHAPE', (11, 1))
gin.constant('environment_metadata.INVERTEDPENDULUM_OBSERVATION_SHAPE', (4, 1))
gin.constant('environment_metadata.REACHER_OBSERVATION_SHAPE', (11, 1))
gin.constant('environment_metadata.SWIMMER_OBSERVATION_SHAPE', (8, 1))
gin.constant('environment_metadata.WALKER2D_OBSERVATION_SHAPE', (17, 1))

''' GYM MUJOCO ACTION SPACE '''
gin.constant('environment_metadata.ANT_ACTION_SHAPE', (8,))
gin.constant('environment_metadata.HALFCHEETAH_ACTION_SHAPE', (6,))
gin.constant('environment_metadata.HOPPER_ACTION_SHAPE', (3,))
gin.constant('environment_metadata.HUMANOID_ACTION_SHAPE', (17,))
gin.constant('environment_metadata.HUMANOIDSTANDUP_ACTION_SHAPE', (17,))
gin.constant('environment_metadata.INVERTEDDOUBLEPENDULUM_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.INVERTEDPENDULUM_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.REACHER_ACTION_SHAPE', (2,))
gin.constant('environment_metadata.SWIMMER_ACTION_SHAPE', (2,))
gin.constant('environment_metadata.WALKER2D_ACTION_SHAPE', (6,))

''' GYM MUJOCO TIME LIMIT '''
gin.constant('environment_metadata.ANT_TIMELIMIT', 1000)
gin.constant('environment_metadata.HALFCHEETAH_TIMELIMIT', 1000)
gin.constant('environment_metadata.HOPPER_TIMELIMIT', 1000)
gin.constant('environment_metadata.HUMANOID_TIMELIMIT', 1000)
gin.constant('environment_metadata.HUMANOIDSTANDUP_TIMELIMIT', 1000)
gin.constant('environment_metadata.INVERTEDDOUBLEPENDULUM_TIMELIMIT', 1000)
gin.constant('environment_metadata.INVERTEDPENDULUM_TIMELIMIT', 1000)
gin.constant('environment_metadata.REACHER_TIMELIMIT', 50)
gin.constant('environment_metadata.SWIMMER_TIMELIMIT', 1000)
gin.constant('environment_metadata.WALKER2D_TIMELIMIT', 1000)

''' GYM MUJOCO ENVIRONMENT VERSION '''
gin.constant('environment_metadata.ANT_ENV_VER', 'v3')
gin.constant('environment_metadata.HALFCHEETAH_ENV_VER', 'v3')
gin.constant('environment_metadata.HOPPER_ENV_VER', 'v3')
gin.constant('environment_metadata.HUMANOID_ENV_VER', 'v3')
gin.constant('environment_metadata.HUMANOIDSTANDUP_ENV_VER', 'v2')
gin.constant('environment_metadata.INVERTEDDOUBLEPENDULUM_ENV_VER', 'v2')
gin.constant('environment_metadata.INVERTEDPENDULUM_ENV_VER', 'v2')
gin.constant('environment_metadata.REACHER_ENV_VER', 'v2')
gin.constant('environment_metadata.SWIMMER_ENV_VER', 'v3')
gin.constant('environment_metadata.WALKER2D_ENV_VER', 'v3')


''' GYM MUJOCO OBSERVATION SPACE '''
gin.constant('environment_metadata.ACROBOT_SWINGUP_OBSERVATION_SHAPE', (6, 1))
gin.constant('environment_metadata.ACROBOT_SWINGUP_SPARSE_OBSERVATION_SHAPE', (6, 1))
gin.constant('environment_metadata.BALL_IN_CUP_CATCH_OBSERVATION_SHAPE', (8, 1))
gin.constant('environment_metadata.CARTPOLE_BALANCE_OBSERVATION_SHAPE', (5, 1))
gin.constant('environment_metadata.CARTPOLE_BALANCE_SPARSE_OBSERVATION_SHAPE', (5, 1))
gin.constant('environment_metadata.CARTPOLE_SWINGUP_OBSERVATION_SHAPE', (5, 1))
gin.constant('environment_metadata.CARTPOLE_SWINGUP_SPARSE_OBSERVATION_SHAPE', (5, 1))
gin.constant('environment_metadata.CARTPOLE_TWO_POLES_OBSERVATION_SHAPE', (8, 1))
gin.constant('environment_metadata.CARTPOLE_THREE_POLES_OBSERVATION_SHAPE', (11, 1))
gin.constant('environment_metadata.CHEETAH_RUN_OBSERVATION_SHAPE', (17, 1))
gin.constant('environment_metadata.FINGER_SPIN_OBSERVATION_SHAPE', (9, 1))
gin.constant('environment_metadata.FINGER_TURN_EASY_OBSERVATION_SHAPE', (12, 1))
gin.constant('environment_metadata.FINGER_TURN_HARD_OBSERVATION_SHAPE', (12, 1))
gin.constant('environment_metadata.FISH_UPRIGHT_OBSERVATION_SHAPE', (21, 1))
gin.constant('environment_metadata.FISH_SWIM_OBSERVATION_SHAPE', (24, 1))
gin.constant('environment_metadata.HOPPER_STAND_OBSERVATION_SHAPE', (15, 1))
gin.constant('environment_metadata.HOPPER_HOP_OBSERVATION_SHAPE', (15, 1))
gin.constant('environment_metadata.HUMANOID_STAND_OBSERVATION_SHAPE', (67, 1))
gin.constant('environment_metadata.HUMANOID_WALK_OBSERVATION_SHAPE', (67, 1))
gin.constant('environment_metadata.HUMANOID_RUN_OBSERVATION_SHAPE', (67, 1))
gin.constant('environment_metadata.HUMANOID_RUN_PURE_STATE_OBSERVATION_SHAPE', (55, 1))
gin.constant('environment_metadata.HUMANOID_CMU_STAND_OBSERVATION_SHAPE', (137, 1))
gin.constant('environment_metadata.HUMANOID_CMU_RUN_OBSERVATION_SHAPE', (137, 1))
gin.constant('environment_metadata.LQR_LQR_2_1_OBSERVATION_SHAPE', (4, 1))
gin.constant('environment_metadata.LQR_LQR_6_2_OBSERVATION_SHAPE', (12, 1))
gin.constant('environment_metadata.MANIPULATOR_BRING_BALL_OBSERVATION_SHAPE', (44, 1))
gin.constant('environment_metadata.MANIPULATOR_BRING_PEG_OBSERVATION_SHAPE', (44, 1))
gin.constant('environment_metadata.MANIPULATOR_INSERT_BALL_OBSERVATION_SHAPE', (44, 1))
gin.constant('environment_metadata.MANIPULATOR_INSERT_PEG_OBSERVATION_SHAPE', (44, 1))
gin.constant('environment_metadata.PENDULUM_SWINGUP_OBSERVATION_SHAPE', (3, 1))
gin.constant('environment_metadata.POINT_MASS_EASY_OBSERVATION_SHAPE', (4, 1))
gin.constant('environment_metadata.POINT_MASS_HARD_OBSERVATION_SHAPE', (4, 1))
gin.constant('environment_metadata.QUADRUPED_WALK_OBSERVATION_SHAPE', (78, 1))
gin.constant('environment_metadata.QUADRUPED_RUN_OBSERVATION_SHAPE', (78, 1))
gin.constant('environment_metadata.QUADRUPED_ESCAPE_OBSERVATION_SHAPE', (101, 1))
gin.constant('environment_metadata.QUADRUPED_FETCH_OBSERVATION_SHAPE', (90, 1))
gin.constant('environment_metadata.REACHER_EASY_OBSERVATION_SHAPE', (6, 1))
gin.constant('environment_metadata.REACHER_HARD_OBSERVATION_SHAPE', (6, 1))
gin.constant('environment_metadata.STACKER_STACK_2_OBSERVATION_SHAPE', (49, 1))
gin.constant('environment_metadata.STACKER_STACK_4_OBSERVATION_SHAPE', (63, 1))
gin.constant('environment_metadata.SWIMMER_SWIMMER6_OBSERVATION_SHAPE', (25, 1))
gin.constant('environment_metadata.SWIMMER_SWIMMER15_OBSERVATION_SHAPE', (61, 1))
gin.constant('environment_metadata.WALKER_STAND_OBSERVATION_SHAPE', (24, 1))
gin.constant('environment_metadata.WALKER_WALK_OBSERVATION_SHAPE', (24, 1))
gin.constant('environment_metadata.WALKER_RUN_OBSERVATION_SHAPE', (24, 1))

''' DMC MUJOCO ACTION SPACE '''
gin.constant('environment_metadata.ACROBOT_SWINGUP_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.ACROBOT_SWINGUP_SPARSE_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.BALL_IN_CUP_CATCH_ACTION_SHAPE', (2,))
gin.constant('environment_metadata.CARTPOLE_BALANCE_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.CARTPOLE_BALANCE_SPARSE_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.CARTPOLE_SWINGUP_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.CARTPOLE_SWINGUP_SPARSE_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.CARTPOLE_TWO_POLES_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.CARTPOLE_THREE_POLES_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.CHEETAH_RUN_ACTION_SHAPE', (6,))
gin.constant('environment_metadata.FINGER_SPIN_ACTION_SHAPE', (2,))
gin.constant('environment_metadata.FINGER_TURN_EASY_ACTION_SHAPE', (2,))
gin.constant('environment_metadata.FINGER_TURN_HARD_ACTION_SHAPE', (2,))
gin.constant('environment_metadata.FISH_UPRIGHT_ACTION_SHAPE', (5,))
gin.constant('environment_metadata.FISH_SWIM_ACTION_SHAPE', (5,))
gin.constant('environment_metadata.HOPPER_STAND_ACTION_SHAPE', (4,))
gin.constant('environment_metadata.HOPPER_HOP_ACTION_SHAPE', (4,))
gin.constant('environment_metadata.HUMANOID_STAND_ACTION_SHAPE', (21,))
gin.constant('environment_metadata.HUMANOID_WALK_ACTION_SHAPE', (21,))
gin.constant('environment_metadata.HUMANOID_RUN_ACTION_SHAPE', (21,))
gin.constant('environment_metadata.HUMANOID_RUN_PURE_STATE_ACTION_SHAPE', (21,))
gin.constant('environment_metadata.HUMANOID_CMU_STAND_ACTION_SHAPE', (56,))
gin.constant('environment_metadata.HUMANOID_CMU_RUN_ACTION_SHAPE', (56,))
gin.constant('environment_metadata.LQR_LQR_2_1_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.LQR_LQR_6_2_ACTION_SHAPE', (2,))
gin.constant('environment_metadata.MANIPULATOR_BRING_BALL_ACTION_SHAPE', (5,))
gin.constant('environment_metadata.MANIPULATOR_BRING_PEG_ACTION_SHAPE', (5,))
gin.constant('environment_metadata.MANIPULATOR_INSERT_BALL_ACTION_SHAPE', (5,))
gin.constant('environment_metadata.MANIPULATOR_INSERT_PEG_ACTION_SHAPE', (5,))
gin.constant('environment_metadata.PENDULUM_SWINGUP_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.POINT_MASS_EASY_ACTION_SHAPE', (2,))
gin.constant('environment_metadata.POINT_MASS_HARD_ACTION_SHAPE', (2,))
gin.constant('environment_metadata.QUADRUPED_WALK_ACTION_SHAPE', (12,))
gin.constant('environment_metadata.QUADRUPED_RUN_ACTION_SHAPE', (12,))
gin.constant('environment_metadata.QUADRUPED_ESCAPE_ACTION_SHAPE', (12,))
gin.constant('environment_metadata.QUADRUPED_FETCH_ACTION_SHAPE', (12,))
gin.constant('environment_metadata.REACHER_EASY_ACTION_SHAPE', (2,))
gin.constant('environment_metadata.REACHER_HARD_ACTION_SHAPE', (2,))
gin.constant('environment_metadata.STACKER_STACK_2_ACTION_SHAPE', (5,))
gin.constant('environment_metadata.STACKER_STACK_4_ACTION_SHAPE', (5,))
gin.constant('environment_metadata.SWIMMER_SWIMMER6_ACTION_SHAPE', (5,))
gin.constant('environment_metadata.SWIMMER_SWIMMER15_ACTION_SHAPE', (14,))
gin.constant('environment_metadata.WALKER_STAND_ACTION_SHAPE', (6,))
gin.constant('environment_metadata.WALKER_WALK_ACTION_SHAPE', (6,))
gin.constant('environment_metadata.WALKER_RUN_ACTION_SHAPE', (6,))

''' DMC MUJOCO TIME LIMIT '''
gin.constant('environment_metadata.ACROBOT_SWINGUP_TIMELIMIT', 1000)
gin.constant('environment_metadata.ACROBOT_SWINGUP_SPARSE_TIMELIMIT', 1000)
gin.constant('environment_metadata.BALL_IN_CUP_CATCH_TIMELIMIT', 1000)
gin.constant('environment_metadata.CARTPOLE_BALANCE_TIMELIMIT', 1000)
gin.constant('environment_metadata.CARTPOLE_BALANCE_SPARSE_TIMELIMIT', 1000)
gin.constant('environment_metadata.CARTPOLE_SWINGUP_TIMELIMIT', 1000)
gin.constant('environment_metadata.CARTPOLE_SWINGUP_SPARSE_TIMELIMIT', 1000)
gin.constant('environment_metadata.CARTPOLE_TWO_POLES_TIMELIMIT', 1000)
gin.constant('environment_metadata.CARTPOLE_THREE_POLES_TIMELIMIT', 1000)
gin.constant('environment_metadata.CHEETAH_RUN_TIMELIMIT', 1000)
gin.constant('environment_metadata.FINGER_SPIN_TIMELIMIT', 1000)
gin.constant('environment_metadata.FINGER_TURN_EASY_TIMELIMIT', 1000)
gin.constant('environment_metadata.FINGER_TURN_HARD_TIMELIMIT', 1000)
gin.constant('environment_metadata.FISH_UPRIGHT_TIMELIMIT', 1000)
gin.constant('environment_metadata.FISH_SWIM_TIMELIMIT', 1000)
gin.constant('environment_metadata.HOPPER_STAND_TIMELIMIT', 1000)
gin.constant('environment_metadata.HOPPER_HOP_TIMELIMIT', 1000)
gin.constant('environment_metadata.HUMANOID_STAND_TIMELIMIT', 1000)
gin.constant('environment_metadata.HUMANOID_WALK_TIMELIMIT', 1000)
gin.constant('environment_metadata.HUMANOID_RUN_TIMELIMIT', 1000)
gin.constant('environment_metadata.HUMANOID_RUN_PURE_STATE_TIMELIMIT', 1000)
gin.constant('environment_metadata.HUMANOID_CMU_STAND_TIMELIMIT', 1000)
gin.constant('environment_metadata.HUMANOID_CMU_RUN_TIMELIMIT', 1000)
gin.constant('environment_metadata.LQR_LQR_2_1_TIMELIMIT', 1000)
gin.constant('environment_metadata.LQR_LQR_6_2_TIMELIMIT', 1000)
gin.constant('environment_metadata.MANIPULATOR_BRING_BALL_TIMELIMIT', 1000)
gin.constant('environment_metadata.MANIPULATOR_BRING_PEG_TIMELIMIT', 1000)
gin.constant('environment_metadata.MANIPULATOR_INSERT_BALL_TIMELIMIT', 1000)
gin.constant('environment_metadata.MANIPULATOR_INSERT_PEG_TIMELIMIT', 1000)
gin.constant('environment_metadata.PENDULUM_SWINGUP_TIMELIMIT', 1000)
gin.constant('environment_metadata.POINT_MASS_EASY_TIMELIMIT', 1000)
gin.constant('environment_metadata.POINT_MASS_HARD_TIMELIMIT', 1000)
gin.constant('environment_metadata.QUADRUPED_WALK_TIMELIMIT', 1000)
gin.constant('environment_metadata.QUADRUPED_RUN_TIMELIMIT', 1000)
gin.constant('environment_metadata.QUADRUPED_ESCAPE_TIMELIMIT', 1000)
gin.constant('environment_metadata.QUADRUPED_FETCH_TIMELIMIT', 1000)
gin.constant('environment_metadata.REACHER_EASY_TIMELIMIT', 1000)
gin.constant('environment_metadata.REACHER_HARD_TIMELIMIT', 1000)
gin.constant('environment_metadata.STACKER_STACK_2_TIMELIMIT', 1000)
gin.constant('environment_metadata.STACKER_STACK_4_TIMELIMIT', 1000)
gin.constant('environment_metadata.SWIMMER_SWIMMER6_TIMELIMIT', 1000)
gin.constant('environment_metadata.SWIMMER_SWIMMER15_TIMELIMIT', 1000)
gin.constant('environment_metadata.WALKER_STAND_TIMELIMIT', 1000)
gin.constant('environment_metadata.WALKER_WALK_TIMELIMIT', 1000)
gin.constant('environment_metadata.WALKER_RUN_TIMELIMIT', 1000)

''' DMC MUJOCO ENVIRONMENT VERSION '''
gin.constant('environment_metadata.ACROBOT_SWINGUP_ENV_VER', None)
gin.constant('environment_metadata.ACROBOT_SWINGUP_SPARSE_ENV_VER', None)
gin.constant('environment_metadata.BALL_IN_CUP_CATCH_ENV_VER', None)
gin.constant('environment_metadata.CARTPOLE_BALANCE_ENV_VER', None)
gin.constant('environment_metadata.CARTPOLE_BALANCE_SPARSE_ENV_VER', None)
gin.constant('environment_metadata.CARTPOLE_SWINGUP_ENV_VER', None)
gin.constant('environment_metadata.CARTPOLE_SWINGUP_SPARSE_ENV_VER', None)
gin.constant('environment_metadata.CARTPOLE_TWO_POLES_ENV_VER', None)
gin.constant('environment_metadata.CARTPOLE_THREE_POLES_ENV_VER', None)
gin.constant('environment_metadata.CHEETAH_RUN_ENV_VER', None)
gin.constant('environment_metadata.FINGER_SPIN_ENV_VER', None)
gin.constant('environment_metadata.FINGER_TURN_EASY_ENV_VER', None)
gin.constant('environment_metadata.FINGER_TURN_HARD_ENV_VER', None)
gin.constant('environment_metadata.FISH_UPRIGHT_ENV_VER', None)
gin.constant('environment_metadata.FISH_SWIM_ENV_VER', None)
gin.constant('environment_metadata.HOPPER_STAND_ENV_VER', None)
gin.constant('environment_metadata.HOPPER_HOP_ENV_VER', None)
gin.constant('environment_metadata.HUMANOID_STAND_ENV_VER', None)
gin.constant('environment_metadata.HUMANOID_WALK_ENV_VER', None)
gin.constant('environment_metadata.HUMANOID_RUN_ENV_VER', None)
gin.constant('environment_metadata.HUMANOID_RUN_PURE_STATE_ENV_VER', None)
gin.constant('environment_metadata.HUMANOID_CMU_STAND_ENV_VER', None)
gin.constant('environment_metadata.HUMANOID_CMU_RUN_ENV_VER', None)
gin.constant('environment_metadata.LQR_LQR_2_1_ENV_VER', None)
gin.constant('environment_metadata.LQR_LQR_6_2_ENV_VER', None)
gin.constant('environment_metadata.MANIPULATOR_BRING_BALL_ENV_VER', None)
gin.constant('environment_metadata.MANIPULATOR_BRING_PEG_ENV_VER', None)
gin.constant('environment_metadata.MANIPULATOR_INSERT_BALL_ENV_VER', None)
gin.constant('environment_metadata.MANIPULATOR_INSERT_PEG_ENV_VER', None)
gin.constant('environment_metadata.PENDULUM_SWINGUP_ENV_VER', None)
gin.constant('environment_metadata.POINT_MASS_EASY_ENV_VER', None)
gin.constant('environment_metadata.POINT_MASS_HARD_ENV_VER', None)
gin.constant('environment_metadata.QUADRUPED_WALK_ENV_VER', None)
gin.constant('environment_metadata.QUADRUPED_RUN_ENV_VER', None)
gin.constant('environment_metadata.QUADRUPED_ESCAPE_ENV_VER', None)
gin.constant('environment_metadata.QUADRUPED_FETCH_ENV_VER', None)
gin.constant('environment_metadata.REACHER_EASY_ENV_VER', None)
gin.constant('environment_metadata.REACHER_HARD_ENV_VER', None)
gin.constant('environment_metadata.STACKER_STACK_2_ENV_VER', None)
gin.constant('environment_metadata.STACKER_STACK_4_ENV_VER', None)
gin.constant('environment_metadata.SWIMMER_SWIMMER6_ENV_VER', None)
gin.constant('environment_metadata.SWIMMER_SWIMMER15_ENV_VER', None)
gin.constant('environment_metadata.WALKER_STAND_ENV_VER', None)
gin.constant('environment_metadata.WALKER_WALK_ENV_VER', None)
gin.constant('environment_metadata.WALKER_RUN_ENV_VER', None)


''' PYBULLET OBSERVATION SPACE '''
gin.constant('environment_metadata.KUKACAMBULLETENV_OBSERVATION_SHAPE', (256, 1))
gin.constant('environment_metadata.THROWERBULLETENV_OBSERVATION_SHAPE', (48, 1))
gin.constant('environment_metadata.HALFCHEETAHBULLETENV_OBSERVATION_SHAPE', (26, 1))
gin.constant('environment_metadata.HUMANOIDBULLETENV_OBSERVATION_SHAPE', (44, 1))
gin.constant('environment_metadata.INVERTEDPENDULUMSWINGUPBULLETENV_OBSERVATION_SHAPE', (5, 1))
gin.constant('environment_metadata.MINITAURBULLETDUCKENV_OBSERVATION_SHAPE', (28, 1))
gin.constant('environment_metadata.INVERTEDPENDULUMBULLETENV_OBSERVATION_SHAPE', (5, 1))
gin.constant('environment_metadata.HUMANOIDFLAGRUNBULLETENV_OBSERVATION_SHAPE', (44, 1))
gin.constant('environment_metadata.HOPPERBULLETENV_OBSERVATION_SHAPE', (15, 1))
gin.constant('environment_metadata.RACECARZEDBULLETENV_OBSERVATION_SHAPE', (10, 1))
gin.constant('environment_metadata.MINITAURBULLETENV_OBSERVATION_SHAPE', (28, 1))
gin.constant('environment_metadata.PUSHERBULLETENV_OBSERVATION_SHAPE', (55, 1))
gin.constant('environment_metadata.WALKER2DBULLETENV_OBSERVATION_SHAPE', (22, 1))
gin.constant('environment_metadata.ANTBULLETENV_OBSERVATION_SHAPE', (28, 1))
gin.constant('environment_metadata.RACECARBULLETENV_OBSERVATION_SHAPE', (2, 1))
gin.constant('environment_metadata.HUMANOIDFLAGRUNHARDERBULLETENV_OBSERVATION_SHAPE', (44, 1))
gin.constant('environment_metadata.INVERTEDDOUBLEPENDULUMBULLETENV_OBSERVATION_SHAPE', (9, 1))
gin.constant('environment_metadata.REACHERBULLETENV_OBSERVATION_SHAPE', (9, 1))
gin.constant('environment_metadata.STRIKERBULLETENV_OBSERVATION_SHAPE', (55, 1))
gin.constant('environment_metadata.KUKABULLETENV_OBSERVATION_SHAPE', (9, 1))
gin.constant('environment_metadata.CARTPOLECONTINUOUSBULLETENV_OBSERVATION_SHAPE', (4, 1))

''' PYBULLET ACTION SPACE '''
gin.constant('environment_metadata.KUKACAMBULLETENV_ACTION_SHAPE', (3,))
gin.constant('environment_metadata.THROWERBULLETENV_ACTION_SHAPE', (7,))
gin.constant('environment_metadata.HALFCHEETAHBULLETENV_ACTION_SHAPE', (6,))
gin.constant('environment_metadata.HUMANOIDBULLETENV_ACTION_SHAPE', (17,))
gin.constant('environment_metadata.INVERTEDPENDULUMSWINGUPBULLETENV_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.MINITAURBULLETDUCKENV_ACTION_SHAPE', (8,))
gin.constant('environment_metadata.INVERTEDPENDULUMBULLETENV_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.HUMANOIDFLAGRUNBULLETENV_ACTION_SHAPE', (17,))
gin.constant('environment_metadata.HOPPERBULLETENV_ACTION_SHAPE', (3,))
gin.constant('environment_metadata.RACECARZEDBULLETENV_ACTION_SHAPE', (2,))
gin.constant('environment_metadata.MINITAURBULLETENV_ACTION_SHAPE', (8,))
gin.constant('environment_metadata.PUSHERBULLETENV_ACTION_SHAPE', (7,))
gin.constant('environment_metadata.WALKER2DBULLETENV_ACTION_SHAPE', (6,))
gin.constant('environment_metadata.ANTBULLETENV_ACTION_SHAPE', (8,))
gin.constant('environment_metadata.RACECARBULLETENV_ACTION_SHAPE', (2,))
gin.constant('environment_metadata.HUMANOIDFLAGRUNHARDERBULLETENV_ACTION_SHAPE', (17,))
gin.constant('environment_metadata.INVERTEDDOUBLEPENDULUMBULLETENV_ACTION_SHAPE', (1,))
gin.constant('environment_metadata.REACHERBULLETENV_ACTION_SHAPE', (2,))
gin.constant('environment_metadata.STRIKERBULLETENV_ACTION_SHAPE', (7,))
gin.constant('environment_metadata.KUKABULLETENV_ACTION_SHAPE', (3,))
gin.constant('environment_metadata.CARTPOLECONTINUOUSBULLETENV_ACTION_SHAPE', (1,))

''' PYBULLET TIME LIMIT '''
gin.constant('environment_metadata.KUKACAMBULLETENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.THROWERBULLETENV_TIMELIMIT', 100)
gin.constant('environment_metadata.HALFCHEETAHBULLETENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.HUMANOIDBULLETENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.INVERTEDPENDULUMSWINGUPBULLETENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.MINITAURBULLETDUCKENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.INVERTEDPENDULUMBULLETENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.HUMANOIDFLAGRUNBULLETENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.HOPPERBULLETENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.RACECARZEDBULLETENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.MINITAURBULLETENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.PUSHERBULLETENV_TIMELIMIT', 150)
gin.constant('environment_metadata.WALKER2DBULLETENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.ANTBULLETENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.RACECARBULLETENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.HUMANOIDFLAGRUNHARDERBULLETENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.INVERTEDDOUBLEPENDULUMBULLETENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.REACHERBULLETENV_TIMELIMIT', 150)
gin.constant('environment_metadata.STRIKERBULLETENV_TIMELIMIT', 100)
gin.constant('environment_metadata.KUKABULLETENV_TIMELIMIT', 1000)
gin.constant('environment_metadata.CARTPOLECONTINUOUSBULLETENV_TIMELIMIT', 200)

''' PYBULLET ENVIRONMENT VERSION '''
gin.constant('environment_metadata.KUKACAMBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.THROWERBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.HALFCHEETAHBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.HUMANOIDBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.INVERTEDPENDULUMSWINGUPBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.MINITAURBULLETDUCKENV_ENV_VER', 'v0')
gin.constant('environment_metadata.INVERTEDPENDULUMBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.HUMANOIDFLAGRUNBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.HOPPERBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.RACECARZEDBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.MINITAURBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.PUSHERBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.WALKER2DBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.ANTBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.RACECARBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.HUMANOIDFLAGRUNHARDERBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.INVERTEDDOUBLEPENDULUMBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.REACHERBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.STRIKERBULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.KUKABULLETENV_ENV_VER', 'v0')
gin.constant('environment_metadata.CARTPOLECONTINUOUSBULLETENV_ENV_VER', 'v0')