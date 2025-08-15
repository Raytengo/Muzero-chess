from pettingzoo.classic import chess_v6  
env = chess_v6.env()  
env.reset()  
print(env.observation_spaces, env.action_spaces)  
