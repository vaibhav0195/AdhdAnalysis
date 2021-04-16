import praw
import time
import json
import requests
import random
class SubredditHelper:

    def __init__(self,clientId,secretKey,userName,pwd,appName,maxData=10000):
        self._clientId = clientId
        self._secretKey = secretKey
        self._userName = userName
        self.pwd = pwd
        self.appName = appName
        self.maxDataLength = maxData
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

    def getPushshiftData(self,before, after, sub):
        url = (
                "https://api.pushshift.io/reddit/search/submission?&size=1000&after="
                + str(after)
                + "&subreddit="
                + str(sub)
                + "&before="
                + str(before)
        )
        iter = 0
        stopLoop = False
        returnData = []
        while not stopLoop:
            iter += 1
            time.sleep(3)
            req = requests.get(url)
            # allow 5 fails before exiting
            if len(returnData) == self.maxDataLength:
                stopLoop = True
            try:
                data = json.loads(req.text)['data']
                for postData in data:
                    author = postData['author']
                    postBody = postData['selftext']
                    title = postData['title']
                    if (" i " in postBody or ' we ' in postBody or "diag" in postBody):
                        returnData.append([author,title,postBody])
            except Exception as e:
                breakpoint()

        return returnData

    def getData(self,before,after,subredit):
        import datetime
        afterDttime = datetime.datetime.fromtimestamp(after)
        currentDttime = afterDttime + datetime.timedelta(days=150)
        stopLoop = False
        returnData = []
        days = 150
        while not stopLoop:
            url = (
                    "https://api.pushshift.io/reddit/search/submission?&size=1000&after="
                    + str(int(after))
                    + "&subreddit="
                    + str(subredit)
                    + "&before="
                    + str(int(currentDttime.timestamp()))
            )
            time.sleep(3)
            req = requests.get(url)
            # allow 5 fails before exiting
            if len(returnData) == self.maxDataLength or currentDttime.timestamp() > before:
                stopLoop = True
            try:
                data = json.loads(req.text)['data']
                for postData in data:
                    author = postData['author']
                    postBody = postData['selftext']
                    title = postData['title']
                    if (" i " in postBody or ' we ' in postBody or "diag" in postBody):
                        returnData.append([author, title, postBody])
                afterDttime = currentDttime
                after = afterDttime.timestamp()
                if len(returnData) > 1:
                    days=30
                currentDttime = afterDttime + datetime.timedelta(days=days)
            except Exception as e:
                print(e)
                break
        return returnData

    def getAnotherData(self,before,after,subredit):
        import datetime
        afterDttime = datetime.datetime.fromtimestamp(after)
        currentDttime = afterDttime + datetime.timedelta(days=150)
        stopLoop = False
        returnData = []
        days = 150
        while not stopLoop:
            url = (
                    "https://api.pushshift.io/reddit/search/submission?&size=1000&after="
                    + str(int(after))
                    + "&subreddit="
                    + str(subredit)
                    + "&before="
                    + str(int(currentDttime.timestamp()))
            )
            time.sleep(3)
            req = requests.get(url)
            # allow 5 fails before exiting
            if len(returnData) == self.maxDataLength or currentDttime.timestamp() > before:
                stopLoop = True
            try:
                data = json.loads(req.text)['data']
                for postData in data:
                    author = postData['author']
                    postBody = postData['selftext']
                    title = postData['title']
                    if (" i " in postBody or ' we ' in postBody or "diag" in postBody):
                        returnData.append([author, title, postBody])
                afterDttime = currentDttime
                after = afterDttime.timestamp()
                if len(returnData) > 1:
                    days=30
                currentDttime = afterDttime + datetime.timedelta(days=days)
            except Exception as e:
                print(e)
                break
        return returnData

if __name__ == '__main__':
    reditConf = {'clientId':'quTJHE1DMsJr2w','secretKey':'pjDXEP7pM2yJgNsnTPkAQyneDe9Yhg',
                 'userName':'zenitsu0195','pwd':'devilmaynotcry','appName':'redditSentimentAnalysis'}
    # adhdOutput = open("adhd.txt","w")
    adhdPartnerOutput = open("adhdPart.txt","w")
    # adhdOutput.write("author;;;title;;;text\n")
    adhdPartnerOutput.write("author;;;title;;;text\n")
    reditHelperObj = SubredditHelper(**reditConf)
    # reditHelperObj.getSubreditData('ADHD')
    count = 0
    total =0
    # data = reditHelperObj.getData(1616502821,1100000000,'ADHD')
    data = reditHelperObj.getAnotherData(1616502821,1100000000,'ADHD_partners')
    random.shuffle(data)
    numberofLines = 2114
    for postData in data:
        if count > numberofLines:
            break
        subText = postData[2]
        author = postData[0]
        title = postData[1]
        subTextNew = subText.replace("\n", " ")
        titleNew = title.replace("\n", " ")
        adhdPartnerOutput.write("{};;;{};;;{}\n".format(author, titleNew, subTextNew))
        count +=1

    # for subreddit in reditHelperObj.getSubreditData(subreditName='ADHD',limit=5000):
    #     total +=1
    #     # print(subreddit.title,subreddit.selftext)
    #     subText = subreddit.selftext
    #     author = subreddit.author
    #     title = subreddit.title
    #     if (" i " in subText or ' we ' in subText or "diag" in subText): #and
    #         print(subText)
    #         subTextNew = subText.replace("\n"," ")
    #         titleNew = title.replace("\n"," ")
    #         print('*********************************')
    #         adhdOutput.write("{};;;{};;;{}\n".format(author,titleNew,subTextNew))
    #         count +=1

    # for subreddit in reditHelperObj.getSubreditData(subreditName='ADHD_partners',limit=count):
    #     subText = subreddit.selftext
    #     author = subreddit.author
    #     title = subreddit.title
    #     subTextNew = subText.replace("\n", " ")
    #     titleNew = title.replace("\n", " ")
    #     adhdPartnerOutput.write("{};;;{};;;{}\n".format(author, titleNew, subTextNew))
    #     print(subreddit.title)
    # print(count,total)
    adhdPartnerOutput.close()
    # adhdOutput.close()
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