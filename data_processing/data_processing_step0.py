import requests
from bs4 import BeautifulSoup
import json

# the subforum lookup table for bladder cancer forum at:
subforums = ['3-newly-diagnosed','8-non-invasive-superficial-bladder-cancer-questions-and-comments','5-muscle-invasive-bladder-cancer-questions-and-comments',
             '7-metastatic-bladder-cancer', '10-women-and-bladder-cancer', '6-men-and-bladder-cancer', '20-caregivers-questions-and-comments',
             '31-articles-of-interest', '17-chit-chat']

# subforums = ['3-newly-diagnosed']

main_site_url = 'https://bladdercancersupport.org'
user_info = {}

for subforum in subforums:

    #########################################################
    # 1. enter a sub-forum, get the content
    #########################################################
    subforum_url = main_site_url + '/forum/' + subforum + '.html'
    print('requesting access to url:', subforum_url)
    r = requests.get(subforum_url)
    # print(r.text)
    forum_soup = BeautifulSoup(r.text, 'html.parser')

    # find the end page to stop the program: <a class="hasTooltip" href="/forum/3-newly-diagnosed.html?start=2000" title="Page:End">End</a>
    end_page_herf = forum_soup.find('a', {'title': 'Page:End'})
    end_page_link = main_site_url + end_page_herf.get('href')

    print('last_page_link is:', end_page_link )

    subforum_info=[]

    start_page = 0

    while True:
        payload = {'start': str(start_page)}
        rr = requests.get(subforum_url, params=payload, headers='')
        # print(rr.text)
        page_soup = BeautifulSoup(rr.text, 'html.parser')
        print('requesting access to url:', rr.url)

        #########################################################
        # 2. given a sub-forum, get the post url
        #########################################################
        for tr in page_soup.find_all('tr', {'class': 'category'}):
            post_herf = tr.find('a')
            post_link = post_herf.get('href')
            final_url = main_site_url + post_link
            print('requesting access to url:', final_url)
            rrr = requests.get(final_url)
            # print(rr.text)
            post_soup = BeautifulSoup(rrr.text, 'html.parser')

            # 1. finding the general info about the post: <script type="application/ld+json">
            post_content = post_soup.find('script', {'type': 'application/ld+json'})

            if post_content is None:
                continue

            post_info = json.loads(post_content.contents[0].string)

            # 2. extracting the replies: <div class="kmessage">
            replies = []

            for reply in post_soup.find_all('div', {'class': 'col-md-10 message-published'}): # <div class="col-md-10 message-published">
                reply_info = {}
                # print(reply)
                message = reply.find('div', {'class': 'kmessage'})

                # Replied by <em><span class="kwho-globalmoderator hasTooltip">Alan</span></em>
                author_id = message.find('span').text.replace('/', '-')
                # print(author_id)
                # <div class="kmsg">
                reply_info['text'] = message.find('div', {'class': 'kmsg'}).text
                # add the user into user_info dict, maintain the signature also there
                if author_id not in user_info:
                    user_info[author_id]={}

                    # get user signature <span class="ksignature">
                    temp_sig = reply.find('span', {'class': 'ksignature'})
                    if temp_sig:
                        sig = temp_sig.text
                    else:
                        sig = None

                    user_info[author_id]['signature'] = sig
                    user_info[author_id]['posts'] = []

                replies.append(reply_info)

                # print( reply_info['author'] + ': #post -> ' + str(user_info[reply_info['author']]['num_posts']))
                reply_info['date'] = post_info['datePublished']
                reply_info['topic'] = post_info['headline']
                reply_info['subforum'] = subforum
                user_info[author_id]['posts'].append(reply_info)

            post_info['posts']= replies

            subforum_info.append(post_info)

        start_page+=20

        # make temp saving
        print('saving current data....')
        with open( 'forum/' + subforum + '.json', 'w') as outfile:
            json.dump(subforum_info, outfile)

        for user in user_info:
            with open('user/' + user + '.json', 'w') as outfile:
                json.dump(user_info[user], outfile)

        # check termination
        if rr.url == end_page_link:
            break
#
# for link in soup.find_all('adenocarcinoma 4 cm in bladder'):
#     print(link.get('href'))
#
# print(r.url)
