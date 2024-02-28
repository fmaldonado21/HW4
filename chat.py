import requests
import streamlit as st
from ibm_watson_machine_learning.foundation_models import Model
import os


url = "https://us-south.ml.cloud.ibm.com"

st.subheader("Enter your API Key")
api_key = st.text_input("Your API Key: ")

def get_credentials(url, api_key):
  return {
    "url" : url,
    "apikey" : api_key
  }


input =  """You are a fortnite bot, your job is to teach new fortnite players how to play fortnite

These are a few rules on how to play. You are going to use this article to answer any fortnite questions

1. You can’t take the things you pick up on Spawn Island with you
When you spawn into Spawn Island you’ll see blasters and building materials scattered around just waiting to be grabbed. And you can grab them - but they won’t come with you into the Fortnite map. Treat Spawn Island as a practice area for how to play Fortnite, although you’ll probably only be there for a minute at most while the game fills up with 100 players. You can launch anyone in the vicinity but the blasters won’t do any damage, or if you’re being a temporary pacifist, try your hand at building a quick sniping hut (more details on how to do that below in point 8). 

2. Wait as long as possible to drop from the Battle Bus
The Battle Bus’ horn will honk as soon as you’re able to drop from it onto the map below, but resist the urge to join the crowd of players who leap from it asap. Instead wait until there’s about 3 seconds to go before the Battle Bus reaches the end of its flight path, then leap out. You’ll have little to no players to compete with for landing spots, which means there’s a smaller chance of getting gunned in the face in your first two minutes of the match and less competition for loot. Aim towards a house or structure when you’re gliding as there’s a high chance of a chest being inside it, or at the very least some basic loot. Just hit the roof with your pickaxe to get in.

3. Drink small shield potions before glugging a large one
Around the map you’ll find both small and large blue bottles, which give you 25 shield and 50 shield respectively if you use your right trigger to drink them (the same button you use for firing your gun). Once you have 50 or higher shield you can’t consume any more small bottles, so make sure to drink them first so you don’t have them burning up a slot in your inventory. You can chug large shield potions no matter the level your shield is at, so if you only have one of them go ahead: pop the lid, and gulp it down asap.

4. 5. Assault blasters or Sub Machine Blasters are good beginner weapons
As a general rule, stick to assault rifles or Sub Machine Blasters when you’re first figuring out how to play Fortnite. Sniper rifles are useless under 75 metres, so although you’ll want to keep one handy do not use it in close quarters combat unless you absolutely have to. Another thing to bear in mind when you’re figuring out how to play Fortnite is if/when you’re launching someone up close and personal, you’ll want to prioritize your shotblaster. Shotblasters - surprise surprise - do a ton of damage and are good for one-hit elim, so keep one in your hands when you’re exploring houses, basements, or any other small spaces.

5. Pay attention to the rarity scale
Grey blasters are the most common, with green, blue, purple, and gold being the ascending order of rarity. Gold blasters are incredibly powerful, so don’t pass them up if you see them lying around. Hit Up on the D-pad to bring up your inventory, and if you hover over the blasters you’ll be able to see how much damage they do if you’re having trouble deciding which to keep. 

6. Build cover before you heal
While I’m on the subject of building, always, always, always build walls around you before you start to heal. Both healing and drinking shield potions take valuable seconds to consume (up to 10 seconds for a large healing kit), during which time you can’t move around, or do anything really except for twirl the camera. Which means you’re very vulnerable indeed, so those walls will stop any bullets connecting with your face. Walls can also clip into cliffs and hills, so there’s no need to find an empty bit of land.

7. There’s fall damage above three storeys
This isn’t Borderlands: falling from a substantial height (above three storeys, to be precise - that’s three of the standard walls one above the other) will take a chunk of your health, so either build ramps downwards or try to slide down hills or cliffs. 

8. Always try to get the upper ground in a fight
When things start to get sticky and you find yourself in a gunfight with one of the 100 people running around the map, you want to be as high as possible. No, not in that way. Make sure you are above your opponent either by building ramps up into the sky, or by jumping repeatedly (which has the added bonus of making you harder to hit). A good strategy is to build four walls around yourself in a square, then build a ramp up to the wall facing your enemy. This’ll provide you with a good sniping platform that you can retreat down when you’re reloading. Just repeat the building formula upwards (jump to build below your feet) until you’re in a makeshift tower and have the upper hand – and gun, and grenade, and so forth...

9. Keep boom booms, bandages, or shield potions in your inventory
It’s worth having at least one of the above items in your toolbar. boom booms are a great way to destroy some of your opponent’s cover, especially if you aim for the lowest part of their structure. Once the lowest level is destroyed, the rest of it will come tumbling down, leaving them out in the open and ready for your hail of bullets. Extra healing and shields will come in handy for when you’re backed into a corner but aren’t ready to give up, or you can share them with your squadmate if they’re in need of some first aid sharpish. 

10. Crouching minimizes the noise your feet make
Sneaky is as sneaky does, and being stealthy can be a great way to edge yourself closed to that coveted Battle Royale. Keeping yourself hidden is part of the trouble, but if you move while crouched you'\''ll move much more quietly than if you run. It'\''s a very useful trick if you can hear someone nearby, because you can manoeuvre yourself into a better hiding place before they come after you. 

11. You have to buy the Battle Pass to get the majority of the rewards
Although there are some perks available without spending a single dime, if you want to get the best skins and other items you'\''ll need the Battle Pass. It costs 950 V-Bucks, meaning you'\''ll need to buy the smallest V-Bucks bundle, at $9.99 / £7.99. After that it'\''s just a case of playing and ticking off as many quests as you can to start earning yourself some gear. 

12. You can change your skin, stickers, and emotes in the locker
All the skins, banner icons, gliders, harvesting tools, loading screen, contrails, emotes, dances, sprays and more all live in your Locker. You can switch them in and out from here before every match, but you won'\''t be able to change any of it in match, so make sure you'\''re happy with your look and your equipped items before you drop. 

Question: How much is the smallest v-bucks bundle?
Answer: $9.99 US Dollars

Question: What is a good beginner Fortnite blaster?
Answer: Assault Blasters and Sub Machine Blasters

"""

parameters = {
		"decoding_method": "greedy",
		"max_new_tokens": 100,
		"stop_sequences": ["\n\n"],
		"repetition_penalty": 1
}
model_id = "meta-llama/llama-2-70b-chat"
project_id = "2a6278cc-949d-4726-a110-0e10666f3ae4"
space_id = os.getenv("SPACE_ID")




def new_model(model_id, parameters, project_id, space_id):
  model = Model(
    model_id = model_id,
    params = parameters,
    credentials = get_credentials(url, api_key),
    project_id = project_id,
    space_id = space_id,
  )
  return model

try:
  model = new_model(model_id, parameters, project_id, space_id)
except Exception as e:
  st.text("Enter API Key")

try: 
  st.title('Hello! I am the Fortnite Helper Bot. Ask me anything!')
  if 'messages' not in st.session_state:
    st.session_state.messages = []
  for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])
  prompt = st.text_input('Question: ')
  if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    response = model.generate_text(prompt = input + prompt + " Answer: ")
    st.chat_message('Fortnite Helper').markdown(response)
    st.session_state.messages.append({
      'role': 'Fortnite Helper', 'content': response
    })
except Exception as e:
  st.text("Enter API key")