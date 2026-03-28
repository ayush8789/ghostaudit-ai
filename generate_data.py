"""
Demo dataset generator — run once: python generate_data.py
Creates demo_data.csv with 1000 records (150 suspicious)
"""
import pandas as pd, random
from datetime import datetime, timedelta
random.seed(42)

FIRST=['Ramesh','Suresh','Geeta','Sunita','Mohan','Rajan','Priya','Anjali','Vijay','Sanjay',
       'Meera','Kavita','Deepak','Rajesh','Anita','Rekha','Ashok','Vinod','Pushpa','Saroj',
       'Lalita','Kiran','Bharat','Santosh','Usha','Mangala','Dinesh','Prakash','Savita','Leela']
LAST=['Kumar','Singh','Sharma','Verma','Yadav','Gupta','Patel','Mishra','Tiwari','Pandey',
      'Devi','Rani','Das','Nath','Jha','Roy','Ghosh','Mukherjee','Reddy','Iyer']
DIST=['Barpeta','Kamrup','Nagaon','Lakhimpur','Goalpara','Jorhat','Sivasagar','Dhubri',
      'Gorakhpur','Varanasi','Lucknow','Allahabad','Kanpur','Darrang','Sonitpur']
VIL=['Rampur','Krishnanagar','Shantinagar','Gandhi Nagar','Nehru Vihar','Bapunagar','Model Town']
SCH=['PM-KISAN','MGNREGA','PM Awas Yojana','Ayushman Bharat','PMKVY Skill Dev','NSP Scholarship','NSAP Pension']
BANKS=['State Bank of India','Bank of Baroda','Punjab National Bank','Canara Bank','UCO Bank']

def rn(): return ''.join([str(random.randint(0,9)) for _ in range(12)])
def rb(): return ''.join([str(random.randint(0,9)) for _ in range(11)])
def rm(): return '9'+''.join([str(random.randint(0,9)) for _ in range(9)])
def name(): return f"{random.choice(FIRST)} {random.choice(LAST)}"
def dob(lo=25,hi=70): return (datetime.now()-timedelta(days=random.randint(lo*365,hi*365))).strftime('%Y-%m-%d')
def reg(): return (datetime.now()-timedelta(days=random.randint(200,1800))).strftime('%Y-%m-%d')
def amt(s):
    r={'PM-KISAN':(2000,6000),'MGNREGA':(3000,15000),'PM Awas Yojana':(120000,150000),
       'Ayushman Bharat':(5000,50000),'PMKVY Skill Dev':(8000,20000),'NSP Scholarship':(3000,12000),'NSAP Pension':(1800,6000)}
    lo,hi=r.get(s,(2000,15000)); return round(random.uniform(lo,hi),2)

def row(aa=None,ba=None,addr=None,dob_r=(25,70),amt_ov=None,mob=None):
    d=random.choice(DIST); s=random.choice(SCH)
    return {'name':name(),'aadhaar':aa or rn(),'bank_account':ba or rb(),'bank_name':random.choice(BANKS),
            'address':addr or f"House {random.randint(1,999)}, {random.choice(VIL)}, {d}",
            'district':d,'scheme':s,'amount':amt_ov or amt(s),
            'dob':dob(*dob_r),'mobile':mob if mob is not None else rm(),'registration_date':reg()}

records=[]
FA1,FA2=rn(),rn(); FB1,FB2,FB3=rb(),rb(),rb()
BD=random.choice(DIST); BOMB=f"Plot No.1, Near Bus Stand, Main Road, {BD}"

for _ in range(850): records.append(row())                                    # clean
for _ in range(8):   records.append(row(aa=FA1))                              # dup aadhaar A
for _ in range(5):   records.append(row(aa=FA2))                              # dup aadhaar B
for _ in range(18):  records.append(row(ba=FB1))                              # shared bank 1
for _ in range(12):  records.append(row(ba=FB2))                              # shared bank 2
for _ in range(9):   records.append(row(ba=FB3))                              # shared bank 3
for _ in range(15):  records.append(row(dob_r=(90,110)))                      # deceased
for _ in range(25):  records.append(row(addr=BOMB,aa=FA1,ba=FB1))             # address bomb
for a in ['11111111111','99999999999','00000000000','12345678901','11111111111',
          '99999999999','00000000000','12345678901','11111111111','99999999999']:
    records.append(row(ba=a))                                                  # fabricated
for _ in range(8):   records.append(row(amt_ov=round(random.uniform(200000,500000),2)))  # amount anomaly

random.shuffle(records)
df=pd.DataFrame(records); df.index=range(1,len(df)+1); df.index.name='id'
df.to_csv('demo_data.csv')
print(f"Created demo_data.csv — {len(df)} records ({len(df)-850} suspicious)")
