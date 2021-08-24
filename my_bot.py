import discord
import os
import requests
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pickle
from discord_webhook import DiscordWebhook, DiscordEmbed
from discord.utils import get
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

SID = SentimentIntensityAnalyzer()
client = discord.Client()

plt.style.use("dark_background")

FOOTER_URL = 'https://images-ext-1.discordapp.net/external/Rf_zzv89NGIbqJp3SIDFggvj-rDPanwAr1KkZd3KpN8/https/media.discordapp.net/attachments/804749266747785297/875100767768436846/Bild1.png'


class SentimentTracker:
	def __init__(self, load_from_existing = False, interval = 10):
		if load_from_existing:
	  		self._load_from_existing(interval)
		else:
			self.START_TIME = datetime.now()
			self.USER_SENT_DICT = {} # doesnt get cleared
			self.USER_SENT_ID_DICT = {} # doesnt get cleared
			self.SENT_SCORE = [] # doesnt get cleared as its used later on
			self.TIMESTAMPS = [] # doesnt get cleared
			self.COUNT_OF_MESSAGES = [] # doesnt get cleared
			self.CORPUS = {}
			self.CORPUS_ID = {}
			self.ALL_MESSAGES = ['']
			self.INTERVAL = interval
  
	def update(self):
		self._update_mean_sent()
		self._update_user_sent_dict()
		self._pickle()
		self.ALL_MESSAGES = ['']
		self.CORPUS = {}
		self.CORPUS_ID = {}
		self.START_TIME = datetime.now()
  
	def _load_from_existing(self, interval):
	    file = open('sentiment_overall.pickle', "rb")
	    self.SENT_SCORE = pickle.load(file)
	    file.close()
	    file = open('sentiment_user.pickle', "rb")
	    self.USER_SENT_DICT = pickle.load(file)
	    file.close()
	    file = open('timestamps.pickle', "rb")
	    self.TIMESTAMPS = pickle.load(file)
	    file.close()
	    file = open('total_messages.pickle', "rb")
	    self.COUNT_OF_MESSAGES = pickle.load(file)
	    file.close()
	    file = open('sentiment_user_id.pickle', 'rb')
	    self.USER_SENT_ID_DICT = pickle.load(file)
	    file.close()
	    self.CORPUS = {}
	    self.CORPUS_ID = {}
	    self.ALL_MESSAGES = ['']
	    self.START_TIME = datetime.now()
	    self.INTERVAL = interval

	def _get_sent(self, arr):
	    scores = []
	    for sentence in arr:
      		scores.append(SID.polarity_scores(sentence)['compound'])
	    return np.mean(scores)

	def _update_mean_sent(self):
	    mean_sentiment = self._get_sent(self.ALL_MESSAGES)
	    print('\tOverall score last 10 min')
	    print(f'\t{mean_sentiment}\n')
	    # if len(ALL_MESSAGES) == 0 -> sent_score.append(0)
	    self.TIMESTAMPS.append(self.START_TIME - timedelta(hours = 4))
	    self.COUNT_OF_MESSAGES.append(len(self.ALL_MESSAGES))
	    self.SENT_SCORE.append(mean_sentiment)
	    print(len(self.SENT_SCORE))
  
	def _update_user_sent_dict(self):
	    for user in self.CORPUS:
	      # for each sentence in CORPUS[user] -> get sent and append
	      	mean_sent = self._get_sent(self.CORPUS[user])
	      	if user in self.USER_SENT_DICT:
	        	self.USER_SENT_DICT[user].append(mean_sent)
	      	else:
	        	self.USER_SENT_DICT[user] = [mean_sent]
	    for user_id in self.CORPUS_ID:
    	# for each sentence in CORPUS_ID[user] -> get sent and append
      		mean_sent = self._get_sent(self.CORPUS_ID[user_id])
      		if user_id in self.USER_SENT_ID_DICT:
      			self.USER_SENT_ID_DICT[user_id].append(mean_sent)
      		else:
      			self.USER_SENT_ID_DICT[user_id] = [mean_sent]
    
	def _pickle(self):
	    sent_scores = open('sentiment_overall.pickle', 'wb')
	    user_sent_scores = open('sentiment_user.pickle', 'wb')
	    user_id_sent_scores = open('sentiment_user_id.pickle', 'wb')
	    timestamps = open('timestamps.pickle', 'wb')
	    total_messages = open('total_messages.pickle', 'wb')
	    pickle.dump(self.SENT_SCORE, sent_scores)
	    pickle.dump(self.USER_SENT_DICT, user_sent_scores)
	    pickle.dump(self.USER_SENT_ID_DICT, user_id_sent_scores)
	    pickle.dump(self.TIMESTAMPS, timestamps)
	    pickle.dump(self.COUNT_OF_MESSAGES, total_messages)
	    sent_scores.close()
	    user_sent_scores.close()
	    user_id_sent_scores.close()
	    timestamps.close()
	    total_messages.close()


class StockNumberFormatter:

	def __init__(self):
		self.embed_sent = datetime.now() + timedelta(hours = 12000)

	def reformat_embed(self, embed_dict):
	    sku_text = ''
	    store_text = ''
	    stock_number_text = ''
	    total_stock_text = ''
	    for field in embed_dict['fields']:
	    	try: 
	    		if field['name'] == 'SKU':
	    			sku_text = field['value']
	    	except:
	    		pass
	    	try:
	    		if field['name'] == 'Store':
	    			store_text = field['value']
	    	except:
	    		pass
	    	try:
	    		if field['name'] == 'Stock Numbers':
	    			stock_number_text = field['value']
	    	except:
	    		pass
	    	try:
	    		if field['name'] == 'Total Stock':
	    			total_stock_text = field['value']
	    	except:
	    		pass
	    thumbnail_url = embed_dict['thumbnail']['url']
	    embed = discord.Embed(color=0x39a244)
	    embed.set_thumbnail(url=thumbnail_url)

	    embed.add_field(name='**__SKU__**', value = sku_text + '\n\n**__Store__**\n' + store_text + '\n\n**__Stock Numbers__**\n' + stock_number_text + '\n\n**__Total Stock__\n**' + total_stock_text, inline = False)
	 
	    embed.set_footer(text='Soflo Supply', icon_url = FOOTER_URL)
	    return embed


s = SentimentTracker(load_from_existing=False, interval = 10)

snf = StockNumberFormatter()

monitor_command_channel = client.get_channel(741199710026727485)

@client.event
async def on_ready():
  print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):

	if str(message.channel) == 'suggestion-feedback' and str(message.author) != 'Legit Check#9257': #806310517760852028
		# message.author.avatar_url
		embed = discord.Embed(color=0x39a244)
		embed.set_author(name = str(message.author), icon_url = message.author.avatar_url)
		embed.set_footer(text='Soflo Supply', icon_url = FOOTER_URL)
		embed.add_field(name = "**__Suggestion:__**", value = f'*{message.content}*', inline = False)
		embed.add_field(name = '**__We would like to hear your opinion!__**', value = 'Should we add this feature? Please vote below!', inline = False)
		msg = await message.channel.send(embed = embed)
		await message.delete()
		await msg.add_reaction("<:yes:856817783631118356>")
		await msg.add_reaction("<:no:856817825629077524>")

	def plot_chart():
		filename = 'sentiment_chart_all_time.png'
		file = open('sentiment_overall.pickle', "rb")
		y = pickle.load(file)
		file.close()
		file = open('timestamps.pickle', "rb")
		x = pickle.load(file)
		file.close()
		avg_sentiment = np.mean(y)
		plt.plot(x, y, color = 'springgreen')
		plt.ylabel('Sentiment')
		plt.xlabel('Time')
		plt.xticks(rotation = 45)
		plt.title('Soflo Supply Member Sentiment')
		plt.tight_layout()
		plt.savefig(filename)
		plt.cla()
		chart = discord.File(filename)
		return chart, filename, avg_sentiment

	try:
		embeds = message.embeds # return list of embeds
		for embed in embeds:
			embed_dict = embed.to_dict() # it's content of embed in dict
			try:
				print(embed_dict)
				if embed_dict['footer']['text'] == '@Kash Monitors • Stock Numbers':
					new_embed = snf.reformat_embed(embed_dict)
					# await message.channel.send(embed = new_embed)
					channel = client.get_channel(875571294593253446)
					await channel.send(embed = new_embed)
					snf.embed_sent = datetime.now()
					print('sent embed')
			except:
				pass
	except:
		pass

	if (datetime.now() - snf.embed_sent).total_seconds() >= 10:
		channel = client.get_channel(741199710026727485)
		snf.embed_sent = datetime.now() + timedelta(hours = 12000)
		await channel.send('Stock Number Role')
		channel = client.get_channel(875571294593253446)
		# await channel.send("Stock Number Role")
		await channel.send(f'<@&{str(875570014969806888)}>')


	if message.content.startswith('!eng'):
		filename = 'total_engagement.png'
		file = open('total_messages.pickle', "rb")
		y = pickle.load(file)
		file.close()
		file = open('timestamps.pickle', "rb")
		x = pickle.load(file)
		file.close()

		plt.plot(x, y, color = 'springgreen')
		plt.ylabel('# of Messages')
		plt.xlabel('Time')
		plt.xticks(rotation = 45)
		plt.title('Soflo Supply Total Engagement')
		plt.tight_layout()
		plt.savefig(filename)
		plt.cla()
		chart = discord.File(filename)
		embed = discord.Embed(title='Soflo Supply Engagement', color=0x39a244)
		embed.set_image(url=f"attachment://{filename}")
		embed.set_thumbnail(url = FOOTER_URL)
		embed.set_footer(text='Soflo Supply', icon_url = FOOTER_URL)
		await message.channel.send(embed=embed, file=chart)


	if message.content.startswith('!sent'):
	    user = str(message.content).split(' ')[1]
	    if user == 'overall':
	      chart, filename, mean_sent = plot_chart()
	      embed = discord.Embed(title='Soflo Supply Member Sentiment', color=0x39a244)
	      embed.set_image(url=f"attachment://{filename}")
	      embed.add_field(name='**Average Sentiment\n**', value = round(mean_sent, 2), inline = False)
	      embed.set_thumbnail(url = FOOTER_URL)
	      embed.set_footer(text='Soflo Supply', icon_url = FOOTER_URL)
	      await message.channel.send(embed=embed, file=chart)
	      
	    else:
	    	user = user[3:-1]
	    	file = open('sentiment_user_id.pickle', "rb")
	    	instance = pickle.load(file)
	    	file.close()
	    	try:
	    		sent = round(np.mean(instance[user]), 3)
	    		if sent > .25:
	    			string = 'positive'
	    		elif sent < -.25:
	    			string = 'negative'
	    		else:
	    			string = 'neutral'
	    		await message.channel.send(f'<@!{user}> avg sentiment: {sent} - {string}')
	    	except:
	    		await message.channel.send('This person has not sent any messages')

	if message.content.startswith('!vars'):
		# ShoePalace stock number and vars
	    url = message.content.split(' ')[1] + '.js'
	    obj = json.loads(requests.get(url).text)
	    shoe_name = obj['title']
	    var = obj['variants']
	    image = obj['media'][0]['preview_image']['src']
	    price = '$' + str(obj['price'])[:-2] + '.' + str(obj['price'])[-2:] + ' USD'
	    var_str = ''
	    stock_str = ''
	    var_only_str = ''
	    site_name = 'ShoePalace'

	    for size_obj in var:
	    	size = size_obj['title']
	    	variant = size_obj['id']
	    	stock = str(size_obj['inventory_quantity'])
	    	if stock.startswith('-'):
	    		stock = stock[1:]
	    	var_only_str += f'{variant}\n'
	    	var_str += f'{size} : {variant}\n'
	    	stock_str += f'{size} : {stock}\n'
	    embed = discord.Embed(title=shoe_name, url=url, color=0x39a244)
	    embed.set_thumbnail(url=image)

	    embed2 = discord.Embed(title=shoe_name, url=url, color=0x39a244)
	    embed2.set_thumbnail(url=image)

	    embed.add_field(name='**Price\n**', value = price, inline = False)
	    embed.add_field(name='**Variants\n**', value=var_str, inline = False)
	    embed.add_field(name='**All Variants\n**', value = '```\n' + var_only_str + '\n```', inline = False)
	    embed.set_author(name=site_name)
	    embed.set_footer(text='Soflo Supply', icon_url = FOOTER_URL)
	    await message.channel.send(embed=embed)

	    embed2.add_field(name='**Stock Numbers\n**', value=stock_str, inline=False)
	    embed2.set_author(name=site_name)
	    embed2.set_footer(text='Soflo Supply', icon_url = FOOTER_URL)
	        
	    await message.channel.send(embed=embed2)

	def clean_message(msg):
		bad_bot_names = ['wrath', 'phantom', 'ghost', 'villain', 'terminator', 'wrath']
		msg = msg.split(' ')
		print('org - ', end="")
		print(msg)
		new_msg = []
		for word in msg:
			if word.startswith('!') and word == msg[0]:
				return ''
			if word.lower() in bad_bot_names:
				pass
			if word.startswith('*') or word.endswith('*'):
				word = word.replace('*', '')
			if word.startswith('`') or word.endswith('`'):
				word = word.replace('`', '')
			if word.startswith('<') or word.startswith('http') or word.startswith('!') or word.startswith('$') or word.startswith('-'):
				pass
			else:
				new_msg.append(word)
		print('\nclean - ', end="")
		print(new_msg)
		return " ".join(new_msg)

	if message.content:
		message.content = clean_message(message.content)
		if message.content != '':
			now = datetime.now()
			if (now - s.START_TIME).total_seconds() / 60 >= s.INTERVAL:
				print(f'|{now}| Updating with new info')
				s.update()
			else:
				if message.content != s.ALL_MESSAGES[-1]:
					s.ALL_MESSAGES.append(message.content)
					print(f'|{now}| Added new message')
					if str(message.author.id) in s.CORPUS_ID:
						s.CORPUS_ID[str(message.author.id)].append(message.content)
						s.CORPUS[str(message.author)].append(message.content)
					else:
						s.CORPUS_ID[str(message.author.id)] = [message.content]
						s.CORPUS[str(message.author)] = [message.content]

client.run("token")
