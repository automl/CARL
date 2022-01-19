import fastrand
import numpy as np


class Mutations():

    def __init__(self, config):
        self.cfg = config
        self.rng = np.random.RandomState(self.cfg.seed.mutation)

    def no_mutation(self, individual):
        individual.train_log["mutation"] = "no_mutation"
        return individual

    def mutation(self, population):

        mutation_options = []
        mutation_proba = []
        if self.cfg.mutation.no_mutation:
            mutation_options.append(self.no_mutation)
            mutation_proba.append(float(self.cfg.mutation.no_mutation))
        if self.cfg.mutation.architecture:
            mutation_options.append(self.architecture_mutate)
            mutation_proba.append(float(self.cfg.mutation.architecture))
        if self.cfg.mutation.parameters:
            mutation_options.append(self.parameter_mutation)
            mutation_proba.append(float(self.cfg.mutation.parameters))
        if self.cfg.mutation.activation:
            mutation_options.append(self.activation_mutation)
            mutation_proba.append(float(self.cfg.mutation.activation))
        if self.cfg.mutation.rl_hyperparam:
            mutation_options.append(self.rl_hyperparam_mutation)
            mutation_proba.append(float(self.cfg.mutation.rl_hyperparam))

        if len(mutation_options) == 0:
            return population

        mutation_proba = np.array(mutation_proba) / np.sum(mutation_proba)

        mutation_choice = self.rng.choice(mutation_options, len(population), p=mutation_proba)

        mutated_population = []
        for mutation, individual in zip(mutation_choice, population):
            mutated_population.append(mutation(individual))

        return mutated_population

    def rl_hyperparam_mutation(self, individual):

        rl_config = individual.rl_config
        rl_params = self.cfg.mutation.rl_hp_selection
        mutate_param = self.rng.choice(rl_params, 1)[0]

        random_num = self.rng.uniform(0, 1)
        if mutate_param == 'train_frames_fraction':
            if random_num > 0.5:
                setattr(rl_config, mutate_param, min(0.1, max(3.0, getattr(rl_config, mutate_param) * 1.2)))
            else:
                setattr(rl_config, mutate_param, min(0.1, max(3.0, getattr(rl_config, mutate_param) * 0.8)))
        elif mutate_param == 'batch_size':
            if random_num > 0.5:
                setattr(rl_config, mutate_param, min(128, max(8, int(getattr(rl_config, mutate_param) * 1.2))))
            else:
                setattr(rl_config, mutate_param, min(128, max(8, int(getattr(rl_config, mutate_param) * 0.8))))
        elif mutate_param == 'lr_actor':
            if random_num > 0.5:
                setattr(rl_config, mutate_param, min(0.005, max(0.00001, getattr(rl_config, mutate_param) * 1.2)))
            else:
                setattr(rl_config, mutate_param, min(0.005, max(0.00001, getattr(rl_config, mutate_param) * 0.8)))
        elif mutate_param == 'lr_critic':
            if random_num > 0.5:
                setattr(rl_config, mutate_param, min(0.005, max(0.00001, getattr(rl_config, mutate_param) * 1.2)))
            else:
                setattr(rl_config, mutate_param, min(0.005, max(0.00001, getattr(rl_config, mutate_param) * 0.8)))
        elif mutate_param == 'td3_policy_noise':
            if getattr(rl_config, mutate_param):
                setattr(rl_config, mutate_param, False)
            else:
                setattr(rl_config, mutate_param, 0.1)
        elif mutate_param == 'td3_update_freq':
            if random_num > 0.5:
                setattr(rl_config, mutate_param, min(10, max(1, int(getattr(rl_config, mutate_param) + 1))))
            else:
                setattr(rl_config, mutate_param, min(10, max(1, int(getattr(rl_config, mutate_param) - 1))))
        elif mutate_param == 'optimizer':
            opti_selection = ["adam", "adamax", "rmsprop", "sdg"]
            opti_selection.remove(getattr(rl_config, mutate_param))
            opti = self.rng.choice(opti_selection, 1)
            setattr(rl_config, mutate_param, opti)

        individual.train_log["mutation"] = "rl_" + mutate_param
        individual.rl_config = rl_config
        return individual

    def activation_mutation(self, individual):
        individual.actor = self._permutate_activation(individual.actor)
        individual.train_log["mutation"] = "activation"
        return individual

    def _permutate_activation(self, network):

        possible_activations = ['relu', 'elu', 'gelu']
        current_activation = network.mlp_activation
        possible_activations.remove(current_activation)
        new_activation = self.rng.choice(possible_activations, size=1)[0]
        net_dict = network.init_dict
        net_dict['mlp_activation'] = new_activation
        net_dict['cnn_activation'] = new_activation
        new_network = type(network)(**net_dict)
        new_network.load_state_dict(network.state_dict())
        network = new_network

        return network

    def parameter_mutation(self, individual):

        offspring = individual.actor

        offspring.cpu()

        offspring = self.classic_parameter_mutation(offspring)
        individual.train_log["mutation"] = "classic_parameter"

        individual.actor = offspring
        return individual

    def regularize_weight(self, weight, mag):
        if weight > mag: weight = mag
        if weight < -mag: weight = -mag
        return weight

    def classic_parameter_mutation(self, network):
        mut_strength = self.cfg.mutation.mutation_sd
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05

        model_params = network.state_dict()

        potential_keys = []
        for i, key in enumerate(model_params):  # Mutate each param
            if not 'norm' in key:
                W = model_params[key]
                if len(W.shape) == 2:  # Weights, no bias
                    potential_keys.append(key)

        how_many = np.random.randint(1, len(potential_keys) + 1, 1)[0]
        chosen_keys = np.random.choice(potential_keys, how_many, replace=False)

        for key in chosen_keys:
            # References to the variable keys
            W = model_params[key]
            num_weights = W.shape[0] * W.shape[1]
            # Number of mutation instances
            num_mutations = fastrand.pcg32bounded(int(np.ceil(num_mutation_frac * num_weights)))
            for _ in range(num_mutations):
                ind_dim1 = fastrand.pcg32bounded(W.shape[0])
                ind_dim2 = fastrand.pcg32bounded(W.shape[-1])
                random_num = self.rng.uniform(0, 1)

                if random_num < super_mut_prob:  # Super Mutation probability
                    W[ind_dim1, ind_dim2] += self.rng.normal(0, np.abs(super_mut_strength * W[ind_dim1, ind_dim2]))
                elif random_num < reset_prob:  # Reset probability
                    W[ind_dim1, ind_dim2] = self.rng.normal(0, 1)
                else:  # mutauion even normal
                    W[ind_dim1, ind_dim2] += self.rng.normal(0, np.abs(mut_strength * W[ind_dim1, ind_dim2]))

                # Regularization hard limit
                W[ind_dim1, ind_dim2] = self.regularize_weight(W[ind_dim1, ind_dim2], 1000000)
        return network

    def architecture_mutate(self, individual):

        offspring_actor = individual.actor.clone()
        offspring_actor.cpu()

        rand_numb = self.rng.uniform(0, 1)
        if 0 <= rand_numb < 0.1:
            offspring_actor.add_mlp_layer()
            individual.train_log["mutation"] = "architecture_new_mlp_layer"

        elif 0.1 <= rand_numb < 0.2:
            offspring_actor.add_cnn_layer()
            individual.train_log["mutation"] = "architecture_new_cnn_layer"

        elif 0.2 <= rand_numb < 0.3:
            offspring_actor.change_cnn_kernal()
            individual.train_log["mutation"] = "architecture_change_cnn_kernal"
        elif 0.3 <= rand_numb < 0.65:
            offspring_actor.add_cnn_channel()
            individual.train_log["mutation"] = "architecture_add_cnn_channel"
        else:
            offspring_actor.add_mlp_node()
            individual.train_log["mutation"] = "architecture_add_mlp_node"

        individual.actor = offspring_actor

        return individual
