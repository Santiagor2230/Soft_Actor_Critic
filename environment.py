import gym
from gym.wrappers import RecordVideo, RecordEpisodeStatistics, \
  FilterObservation, FlattenObservation
  
  
def create_environment(name):
  env = gym.make(name)
  env = FilterObservation(env,["observation", "desired_goal"]) #two keys for state
  env = FlattenObservation(env) # flatten both array observation and desired goal into one array
  env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda x: x % 50 == 0)
  env = RecordEpisodeStatistics(env)
  return env