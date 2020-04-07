class RandomAgent:

    def __init__(self, env):
        self.env = env
        
        self.actions_cnt = env.action_space.n
        
        self._max_iter = 2
        self._gamma = 0.99
        self._final_reward_weight = 1.0
  

    def predict_action(self, state):
        """
        Return action that should be done from input state according to current policy.
        Args:
            state: list of points - results of raycasting
        return: 
            action: int
        """    
        # some cool RL staff
        return self.env.action_space.sample()


    def evaluate(self):
        """
        Generate CAD model, reconstruct it and count the reward according
        to MSE between original and reconstructed models and number of steps.
        Args:
            environment: Environment
            max_iter: int - max number of iterations to stop (~15)
            gamma: float - discounted factor
            w: float - weight of mse to final episode reward
        return: 
            episode_reward: float
        """    

        state = self.env.reset()
        episode_reward = 0.0
        states, actions = [], []
        for t in range(self._max_iter):
            action = self.predict_action(state)
            actions.append(action)
            state, reward, done, info = self.env.step(action)
            print("STEP: ", t, "REWARD: ", reward)
            states.append(state)
            self.env.render(action, state)
            episode_reward += reward * self._gamma ** t

            if done:
                break

        final_reward = self.env.final_reward()
        print("Hausdorff reward: ", final_reward)
        episode_reward += self._final_reward_weight / final_reward # QUESTION
        return episode_reward, states, actions

