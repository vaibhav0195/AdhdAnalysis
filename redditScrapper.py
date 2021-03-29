import praw

class SubredditHelper:

    def __init__(self,clientId,secretKey,userName,pwd,appName):
        self._clientId = clientId
        self._secretKey = secretKey
        self._userName = userName
        self.pwd = pwd
        self.appName = appName
        self._redit = praw.Reddit(client_id=self._clientId,
                         client_secret=self._secretKey,
                         user_agent=self.appName,
                         username=self._userName,
                     password=self.pwd)

    def getSubreditData(self,subreditName,limit=100):
        subreddit = self._redit.subreddit(subreditName)
        # subreddit.search()
        for submission in subreddit.top(limit=limit):
            yield submission
            # print(submission.title)

if __name__ == '__main__':
    reditConf = {'clientId':'quTJHE1DMsJr2w','secretKey':'pjDXEP7pM2yJgNsnTPkAQyneDe9Yhg',
                 'userName':'zenitsu0195','pwd':'devilmaynotcry','appName':'redditSentimentAnalysis'}
    adhdOutput = open("adhd.txt","w")
    adhdPartnerOutput = open("adhdPart.txt","w")
    adhdOutput.write("author;;;title;;;text\n")
    adhdPartnerOutput.write("author;;;title;;;text\n")
    reditHelperObj = SubredditHelper(**reditConf)
    # reditHelperObj.getSubreditData('ADHD')
    count = 0
    total =0
    for subreddit in reditHelperObj.getSubreditData(subreditName='ADHD',limit=5000):
        total +=1
        # print(subreddit.title,subreddit.selftext)
        subText = subreddit.selftext
        author = subreddit.author
        title = subreddit.title
        if (" i " in subText or ' we ' in subText or "diag" in subText): #and
            print(subText)
            subTextNew = subText.replace("\n"," ")
            titleNew = title.replace("\n"," ")
            print('*********************************')
            adhdOutput.write("{};;;{};;;{}\n".format(author,titleNew,subTextNew))
            count +=1

    for subreddit in reditHelperObj.getSubreditData(subreditName='ADHD_partners',limit=count):
        subText = subreddit.selftext
        author = subreddit.author
        title = subreddit.title
        subTextNew = subText.replace("\n", " ")
        titleNew = title.replace("\n", " ")
        adhdPartnerOutput.write("{};;;{};;;{}\n".format(author, titleNew, subTextNew))
        print(subreddit.title)
    print(count,total)
    adhdPartnerOutput.close()
    adhdOutput.close()
    # chars14 = 'quTJHE1DMsJr2w'
    # secretKey = 'pjDXEP7pM2yJgNsnTPkAQyneDe9Yhg'
    # userName = 'zenitsu0195'
    # pwd = 'devilmaynotcry'
    # limit= 5000000
    # count = 0
    # reddit = praw.Reddit(client_id=chars14,
    #                      client_secret=secretKey,
    #                      user_agent='redditSentimentAnalysis',
    #                      username=userName,
    #                  password=pwd)
    #
    # subreddit = reddit.subreddit('ADHD')
    # listOfUsers = []
    #
    # for submission in subreddit.top():
    #     subTitle = submission.title.lower()
    #     if  (" i " in subTitle or ' we ' in subTitle): #("diag" in subTitle): #and
    #         print(submission.title)
    #         print(submission.author)
    #         listOfUsers.append(submission.author)
    #         count +=1
    # subredditPartner = reddit.subreddit('ADHD_partners')
    # print("number of unique users {}".format(len(list(set(listOfUsers)))))
    # print('got {} entries'.format(count))
    # print('connected')