import collections
import gym
from keras_models import MLP
import numpy as np
import random
import time

# Gym global variables
GAME = 'CartPole-v0'                      # The game we are playing
MODEL_OUTPUT = 'models/MLP_DDQN_'+GAME+'.h5'   # Where to save the models
MAX_EPISODE_STEPS = 10000                 # Number of steps to consider the game is won (default = 200)
gym.envs.registry.env_specs['CartPole-v0'].max_episode_steps = MAX_EPISODE_STEPS

# Q-learning global variables
LEARNING_STEPS          = 50000    # Number of Q-learning steps
STEPS_TO_UPDATE_NETWORK = 500      # Number of Q-learning steps between each update of the Q_network with weights of the target_Q_network
INITIAL_MEMORY_SIZE     = 1000     # Number of frames to save in memory before beginning Q-learning
MAX_MEMORY_SIZE         = 10000    # Maximum number of frames in memory (after we start to pop out old memory)
MINIBATCH_SIZE          = 32       # Size of minibatches on which the Q_network is trained
GAMMA                   = 0.99     # Bellman's equation discount parameter

EPSILON_INI             = 1        # Initial value of epsilon (to decide whether to pick an action at random or not)
EPSILON_MIN             = 0.1      # Minimal value of epsilon
EPSILON_STEPS           = 10000    # Number of Q-learning steps to go from EPSILON_INI to EPSILON_MIN

KERAS_VERBOSE           = False    # Set Keras to verbose mode

RENDER_GAME             = 5000     # Number of steps between each render of a game (set to 0 for no render)
SAVE_STEPS              = 1000     # Number of steps between 2 saves of the model

def epsilon(x):
    '''
    Attenuation function to reduce the fraction of random actions
    '''
    if x < EPSILON_STEPS:
        return EPSILON_INI + (EPSILON_MIN - EPSILON_INI) / EPSILON_STEPS * x 
    else:
        return EPSILON_MIN

def play(game,model,render):
    '''
    Play a gym game
    '''
    print("--------------")
    print("Playing a game")
    env = gym.make(game)
    state = env.reset()

    done = False
    score = 0
    while not done:
        action = np.argmax(model.predict( np.expand_dims(state,axis=0) )[0])
        state, reward, done, info = env.step(action)
        score+= reward
        if render:
            env.render()
    print("Scored:", score)
    print("--------------")

def main():
    time_at_begining = time.time()
    
    # Initialize gym environment
    env = gym.make(GAME)
    env.spec.max_episode_steps = MAX_EPISODE_STEPS
    state = env.reset()

    # Initialize Q_networks
    input_shape = state.shape
    Q_network = MLP(input_shape,env.action_space.n)
    target_Q_network = MLP(input_shape,env.action_space.n)

    # Initialize memmory
    memory = collections.deque()

    done = False
    print("Build up initial memory:")
    while len(memory) < INITIAL_MEMORY_SIZE:
        if len(memory)%100 == 0:
            print("Number of saved states:",len(memory),"/",INITIAL_MEMORY_SIZE)

        if done:
            state = env.reset()

        # Pick a random action
        action = env.action_space.sample()
        # Generate new state
        new_state, reward, done, info = env.step(action)
        # Save state, action, reward, new_state
        memory.append([state, action, reward, new_state, done])
        state = new_state

    score = 0
    scores = []
    number_of_games = 0
    current_game_done = done
    print("Begin Q-learning")
    for _ in range(LEARNING_STEPS+1): # +1 to save the model on the last step
        if _%100 == 0:
            print("Q-learning step",_,"/",LEARNING_STEPS)

        # Reinitialize the env if game is done
        if current_game_done:
            number_of_games+=1
            scores.append(score)
            score = 0
            state = env.reset()

        # Pick an action according to epsilon-greedy policy
        random_number = random.uniform(0,1)
        if random_number < epsilon(_):
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_network.predict( np.expand_dims(state,axis=0) )[0])
        
        # Generate new state
        new_state, reward, done, info = env.step(action)
        score+= reward
        current_game_done = done

        # Save state, action, reward, new_state
        memory.append([state, action, reward, new_state, done])
        state = new_state

        # Remove old memory as we go
        while len(memory) > MAX_MEMORY_SIZE:
            memory.popleft()

        # Sample random minibatch
        recollection = random.sample(memory,MINIBATCH_SIZE)
        minibatch = []
        targets = []
        for event in recollection:
            r_state     = event[0]
            r_action    = event[1]
            r_reward    = event[2]
            r_new_state = event[3]
            r_done      = event[4]
            # Define the targets for the action 
            # using the model prediction for all actions
            # but the one that was choosen which is updated
            # with Bellman's equation (if game is note done).
            target = target_Q_network.predict( np.expand_dims(r_state,axis=0) )[0]
            if r_done:
                target[r_action] = r_reward
            else:
                target[r_action] = r_reward + GAMMA * np.max(Q_network.predict( np.expand_dims(r_new_state,axis=0) )[0])        
            
            # Append lists 
            minibatch.append(r_state)
            targets.append(target)
        
        # transform to arrays as arrays are required for Keras   
        minibatch = np.asarray(minibatch)
        targets = np.asarray(targets)
        
        # Training the Q_network
        Q_network.fit(minibatch,targets,batch_size=MINIBATCH_SIZE,epochs=1,verbose=KERAS_VERBOSE)

        if _%STEPS_TO_UPDATE_NETWORK == 0:
            if number_of_games>0:
                print("---------------")
                print("Game statistics")
                print("---------------")
                print("# games:",number_of_games,"| Mean score:",np.mean(scores),"| Median score:",np.median(scores),"| Min score:",np.min(scores),"| Max score:",np.max(scores),"| Randomness:",epsilon(_))
                print("---------------")
            number_of_games = 0
            scores = []

            # Update target_Q_network with weights of Q_network
            target_Q_network.set_weights(Q_network.get_weights())
        
        if RENDER_GAME>0 and _%RENDER_GAME == 0:
            play(GAME,Q_network,render=True)

        if _%SAVE_STEPS == 0:
            Q_network.save(MODEL_OUTPUT+'_step'+str(_))

    print("Execution time:",time.time() - time_at_begining)       

if __name__ == "__main__":
    main()




        