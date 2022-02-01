if __name__ == "__main__":
    from typing import List

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from carl.utils.doc_building.plotting import radar_factory

    env_context_feature_names = {
        "CARLMountainCarEnv": [
            "force",
            "goal_position",
            "goal_velocity",
            "gravity",
            "max_position",
            "max_speed",
            "min_position",
            "start_position",
            "start_position_std",
            "start_velocity",
            "start_velocity_std",
        ],
        "CARLPendulumEnv": ["dt", "g", "l", "m", "max_speed"],
        "CARLAcrobotEnv": [
            "link_com_1",
            "link_com_2",
            "link_length_1",
            "link_length_2",
            "link_mass_1",
            "link_mass_2",
            "link_moi",
            "max_velocity_1",
            "max_velocity_2",
        ],
        "CARLCartPoleEnv": [
            "force_magnifier",
            "gravity",
            "masscart",
            "masspole",
            "pole_length",
            "update_interval",
        ],
        "CARLMountainCarContinuousEnv": [
            "goal_position",
            "goal_velocity",
            "max_position",
            "max_position_start",
            "max_speed",
            "max_velocity_start",
            "min_position",
            "min_position_start",
            "min_velocity_start",
            "power",
        ],
        "CARLLunarLanderEnv": [
            "FPS",
            "GRAVITY_X",
            "GRAVITY_Y",
            "INITIAL_RANDOM",
            "LEG_AWAY",
            "LEG_DOWN",
            "LEG_H",
            "LEG_SPRING_TORQUE",
            "LEG_W",
            "MAIN_ENGINE_POWER",
            "SCALE",
            "SIDE_ENGINE_AWAY",
            "SIDE_ENGINE_HEIGHT",
            "SIDE_ENGINE_POWER",
            "VIEWPORT_H",
            "VIEWPORT_W",
        ],
        "CARLVehicleRacingEnv": ["VEHICLE"],
        "CARLBipedalWalkerEnv": [
            "FPS",
            "FRICTION",
            "GRAVITY_X",
            "GRAVITY_Y",
            "INITIAL_RANDOM",
            "LEG_DOWN",
            "LEG_H",
            "LEG_W",
            "LIDAR_RANGE",
            "MOTORS_TORQUE",
            "SCALE",
            "SPEED_HIP",
            "SPEED_KNEE",
            "TERRAIN_GRASS",
            "TERRAIN_HEIGHT",
            "TERRAIN_LENGTH",
            "TERRAIN_STARTPAD",
            "TERRAIN_STEP",
            "VIEWPORT_H",
            "VIEWPORT_W",
        ],
        "CARLAnt": [
            "actuator_strength",
            "angular_damping",
            "friction",
            "gravity",
            "joint_angular_damping",
            "joint_stiffness",
            "torso_mass",
        ],
        "CARLHalfcheetah": [
            "angular_damping",
            "friction",
            "gravity",
            "joint_angular_damping",
            "joint_stiffness",
            "torso_mass",
        ],
        "CARLHumanoid": [
            "angular_damping",
            "friction",
            "gravity",
            "joint_angular_damping",
            "torso_mass",
        ],
        "CARLFetch": [
            "actuator_strength",
            "angular_damping",
            "friction",
            "gravity",
            "joint_angular_damping",
            "joint_stiffness",
            "target_distance",
            "target_radius",
            "torso_mass",
        ],
        "CARLGrasp": [
            "actuator_strength",
            "angular_damping",
            "friction",
            "gravity",
            "joint_angular_damping",
            "joint_stiffness",
            "target_distance",
            "target_height",
            "target_radius",
        ],
        "CARLUr5e": [
            "actuator_strength",
            "angular_damping",
            "friction",
            "gravity",
            "joint_angular_damping",
            "joint_stiffness",
            "target_distance",
            "target_radius",
            "torso_mass",
        ],
        "CARLRnaDesignEnv": [
            "mutation_threshold",
            "reward_exponent",
            "state_radius",
            "dataset",
            "target_structure_ids",
        ],
        "CARLMarioEnv": ["level_index", "noise", "mario_state"],
    }
    action_space_sizes = [
        (3,),
        (1,),
        (3,),
        (2,),
        (1,),
        (4,),
        (3,),
        (4,),
        (8,),
        (6,),
        (17,),
        (10,),
        (19,),
        (6,),
        (8,),
        (10,),
    ]
    state_space_sizes = [
        (2,),
        (3,),
        (6,),
        (4,),
        (2,),
        (8,),
        (96, 96, 3),
        (24,),
        (87,),
        (23,),
        (299,),
        (101,),
        (132,),
        (66,),
        (11,),
        (64, 64, 3),
    ]
    n_context_features = [11, 5, 9, 6, 10, 16, 1, 20, 7, 6, 5, 9, 9, 9, 5, 3]
    env_names = [
        "CARLMountainCarEnv",
        "CARLPendulumEnv",
        "CARLAcrobotEnv",
        "CARLCartPoleEnv",
        "CARLMountainCarContinuousEnv",
        "CARLLunarLanderEnv",
        "CARLVehicleRacingEnv",
        "CARLBipedalWalkerEnv",
        "CARLAnt",
        "CARLHalfcheetah",
        "CARLHumanoid",
        "CARLFetch",
        "CARLGrasp",
        "CARLUr5e",
        "CARLRnaDesignEnv",
        "CARLMarioEnv",
    ]
    n_cfs_d = [11, 5, 8, 6, 10, 16, 1, 20, 7, 6, 5, 9, 9, 9, 4, 3]
    n_cfs_r = [0, 0, 0, 0, 0, 4, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0]
    n_cfs = 131
    n_dynami_changing = 129
    n_reward_changing = 7
    n_float_cfs = 114
    percentage_float_cfs = n_float_cfs / n_cfs

    env_types = {
        "classic_control": [
            "CARLAcrobotEnv",
            "CARLCartPoleEnv",
            "CARLMountainCarEnv",
            "CARLMountainCarContinuousEnv",
            "CARLPendulumEnv",
        ],
        "box2d": ["CARLBipedalWalkerEnv", "CARLLunarLanderEnv", "CARLVehicleRacingEnv"],
        "brax": ["CARLAnt", "CARLFetch", "CARLGrasp", "CARLHumanoid", "CARLUr5e"],
        "misc": ["CARLMarioEnv", "CARLRnaDesignEnv"],
    }

    data = []  # type: List[pd.DataFrame]
    for env_type in env_types:
        envs = env_types[env_type]

        title = env_type

        ids = [env_names.index(e) for e in envs]
        # ss_sizes = [state_space_sizes[i][0] for i in ids]
        # as_sizes = [action_space_sizes[i][0] for i in ids]
        ss_sizes = [np.prod(state_space_sizes[i]) for i in ids]
        as_sizes = [np.prod(action_space_sizes[i]) for i in ids]
        reward_changing = [n_cfs_r[i] for i in ids]
        dynamics_changing = [n_cfs_d[i] for i in ids]
        cf_numbers = [len(env_context_feature_names[env_names[i]]) for i in ids]
        # print(ss_sizes, as_sizes, cf_numbers)
        data.append(
            pd.DataFrame(
                {
                    "env_type": [env_type] * len(ids),
                    "env_name": envs,
                    "state_space_size": ss_sizes,
                    "action_space_size": as_sizes,
                    "n_context_features": cf_numbers,
                    "n_cf_reward": reward_changing,
                    "n_cf_dyna": dynamics_changing,
                }
            )
        )
    data = pd.concat(data)

    # normalize values
    cols = [c for c in data.columns if c not in ["env_type", "env_name"]]
    max_values_per_col = []
    for col in cols:
        if col == "state_space_size":
            data[col] = np.log(data[col])
        max_val = data[col].max()
        max_values_per_col.append(max_val)
        data[col] /= max_val

    cols_plot = [
        "state_space_size",
        "action_space_size",
        "n_cf_reward",
        "n_cf_dyna",
        "n_context_features",
    ]
    xticklabels = [
        "state space size",
        "action\nspace \nsize",
        "$n_{cf, reward}$",
        "$n_{cf,dynamics}$",
        "$n_{cf}$",
    ]

    figtitle = "Environments"
    N = len(cols_plot)
    theta = radar_factory(N, frame="polygon")

    figsize = (10, 2.5)
    dpi = 250
    fig, axs = plt.subplots(
        figsize=figsize, nrows=1, ncols=4, subplot_kw=dict(projection="radar"), dpi=dpi
    )
    # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.99, bottom=0.01)

    # Plot the four cases from the example data on separate axes
    for ax, env_type in zip(axs.flat, env_types):
        D = data[data["env_type"] == env_type]
        labels = D["env_name"].to_list()
        color_palette_name = "colorblind"
        n = len(D)
        colors = sns.color_palette(color_palette_name, n)

        plot_data = D[cols_plot].to_numpy()

        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        title = env_type.replace("_", " ")
        if title == "misc":
            title = "RNA + Mario"
        ax.set_title(
            title,
            weight="normal",
            size="medium",  # position=(0.5, 0.25), transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            pad=15,
            fontsize=12,
        )
        for i, (d, color) in enumerate(zip(plot_data, colors)):
            ax.plot(theta, d, color=color, label=labels[i])
            ax.fill(theta, d, facecolor=color, alpha=0.25)
            ax.set_varlabels(
                xticklabels, horizontalalignment="center", verticalalignment="center"
            )
        # ax.legend(loc=(0.25, -.5), labelspacing=0.1, fontsize='small')
        rticks = np.linspace(0, 1, 5)
        ax.set_rticks(rticks)
        plt.setp(ax.get_yticklabels(), visible=False)

    # add legend relative to top-left plot
    # labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
    # legend = axs[0, 0].legend(labels, loc=(0.9, .95),
    #                           labelspacing=0.1, fontsize='small')

    # fig.text(0.5, 0.965, figtitle,
    #          horizontalalignment='center', color='black', weight='bold',
    #          size='large')
    fig.set_tight_layout(True)

    figfname = "utils/radar_env_space.png"
    fig.savefig(figfname, bbox_inches="tight")
    plt.show()
