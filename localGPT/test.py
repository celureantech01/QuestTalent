import pdb

pdb.set_trace()


from proxycurl.asyncio import Proxycurl
import asyncio

proxycurl = Proxycurl()
person = asyncio.run(proxycurl.linkedin.person.get(
    linkedin_profile_url='https://www.linkedin.com/in/yadavendra-yadav-2b61b122/',
    extra='include'
))
print('Person Result:', person)

