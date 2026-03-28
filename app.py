"""
GhostAudit AI - Backend
Run: python app.py
Then open: http://localhost:5000
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd, numpy as np, os, json, urllib.request, urllib.error
from collections import defaultdict
from datetime import datetime

app = Flask(__name__, static_folder='frontend')
CORS(app)

# ── Read keys directly from .env file or environment ─────────────
def load_key(name, env_var=None):
    try:
        with open('.env') as f:
            for line in f:
                line = line.strip()
                if line.startswith(name + '='):
                    val = line.split('=', 1)[1].strip().strip('"').strip("'")
                    if val and val != 'your_key_here': return val
    except: pass
    # Then try environment variable
    val = os.environ.get(name, '').strip()
    return val if val and val != 'your_key_here' else ''

GROQ_KEY      = load_key('GROQ_KEY')
GEMINI_KEY    = load_key('GEMINI_KEY')
OPENROUTER_KEY = load_key('OPENROUTER_KEY', 'OPENROUTER_KEY')
ANTHROPIC_KEY = load_key('ANTHROPIC_KEY')

def get_provider():
    if OPENROUTER_KEY: return 'openrouter'
    if GROQ_KEY:       return 'groq'
    if GEMINI_KEY:     return 'gemini'
    if ANTHROPIC_KEY:  return 'anthropic'
    return 'none'

# ── AI Calls ─────────────────────────────────────────────────────
def ai_groq(prompt):
    # Try models in order — first one that works is used
    models = [
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    last_err = None
    for model in models:
        try:
            data = json.dumps({
                "model": model,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_KEY}"}
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                result = json.loads(r.read())['choices'][0]['message']['content']
                print(f"[Groq] Using model: {model}")
                return result
        except urllib.error.HTTPError as e:
            print(f"[Groq] {model} failed: {e.code} — trying next...")
            last_err = e
            continue
    raise last_err

def ai_gemini(prompt):
    models = ["gemini-2.0-flash","gemini-1.5-flash","gemini-1.5-flash-latest","gemini-1.0-pro"]
    data = json.dumps({"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"maxOutputTokens":500}}).encode()
    last_err = None
    for model in models:
        try:
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}",
                data=data, headers={"Content-Type":"application/json"}
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                result = json.loads(r.read())['candidates'][0]['content']['parts'][0]['text']
                print(f"[Gemini] Using model: {model}")
                return result
        except urllib.error.HTTPError as e:
            print(f"[Gemini] {model} failed: {e.code} — trying next...")
            last_err = e
            continue
    raise last_err

def ai_openrouter(prompt):
    models = [
        "meta-llama/llama-3.1-8b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "google/gemma-2-9b-it:free",
    ]
    last_err = None
    for model in models:
        try:
            data = json.dumps({
                "model": model,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "HTTP-Referer": "http://localhost:5000",
                }
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                result = json.loads(r.read())['choices'][0]['message']['content']
                print(f"[OpenRouter] Using model: {model}")
                return result
        except urllib.error.HTTPError as e:
            print(f"[OpenRouter] {model} failed: {e.code} — trying next...")
            last_err = e
            continue
    raise last_err

def ai_anthropic(prompt):
    data = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={"Content-Type": "application/json",
                 "x-api-key": ANTHROPIC_KEY,
                 "anthropic-version": "2023-06-01"}
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())['content'][0]['text']

def call_ai(prompt):
    p = get_provider()
    if p == 'openrouter': return ai_openrouter(prompt), 'openrouter'
    if p == 'groq':       return ai_groq(prompt), 'groq'
    if p == 'gemini':     return ai_gemini(prompt), 'gemini'
    if p == 'anthropic':  return ai_anthropic(prompt), 'anthropic'
    return None, 'none'

_or_free_models = []

def _fetch_free_models():
    global _or_free_models
    if _or_free_models: return _or_free_models
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        free = [m["id"] for m in data.get("data",[])
                if str(m.get("pricing",{}).get("completion","1")) == "0"
                or m["id"].endswith(":free")]
        _or_free_models = free[:6]
        print(f"[OpenRouter] Found {len(_or_free_models)} free models: {_or_free_models[:3]}")
        return _or_free_models
    except Exception as e:
        print(f"[OpenRouter] Could not fetch models: {e}")
        _or_free_models = [
            "meta-llama/llama-3.2-3b-instruct:free",
            "qwen/qwen-2.5-7b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "google/gemma-3-4b-it:free",
            "deepseek/deepseek-r1-distill-qwen-1.5b:free",
        ]
        return _or_free_models

def ai_openrouter(prompt):
    models = _fetch_free_models()
    last_err = None
    for model in models:
        try:
            data = json.dumps({
                "model": model, "max_tokens": 500,
                "messages": [{"role":"user","content":prompt}]
            }).encode()
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions", data=data,
                headers={"Content-Type":"application/json",
                         "Authorization":f"Bearer {OPENROUTER_KEY}",
                         "HTTP-Referer":"http://localhost:5000",
                         "X-Title":"GhostAudit AI"}
            )
            with urllib.request.urlopen(req, timeout=25) as r:
                resp_json = json.loads(r.read())
                try:
                    result = resp_json['choices'][0]['message']['content']
                    if result is None or not isinstance(result, str) or not result.strip():
                        raise ValueError(f"Empty or invalid content from {model}")
                except (KeyError, IndexError, TypeError):
                    raise ValueError(f"Unexpected OpenRouter response format: {resp_json}")
                print(f"[OpenRouter] OK: {model}")
                return result
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:120]
            print(f"[OpenRouter] {model} failed {e.code}: {body}")
            last_err = e
            continue
        except Exception as e:
            print(f"[OpenRouter] {model} error: {e}")
            last_err = e
            continue
    raise last_err or Exception("All OpenRouter models failed")

def smart_explanation(case):
    """Detailed explanation without API - always works"""
    details = {d['type']: d for d in case.get('risk_details', [])}
    amt = case.get('amount', 0)
    name = case.get('name', 'Unknown')

    if 'FABRICATED_BANK' in details and 'DUPLICATE_AADHAAR' in details:
        return {
            'fraud_type': 'Coordinated Identity Ring',
            'assessment': f"{name} shows both a fabricated bank account AND duplicate Aadhaar across {details['DUPLICATE_AADHAAR'].get('count',2)} beneficiaries — a hallmark of organised fraud rings documented in CAG's PMKVY 2025 audit.",
            'severity': 'CRITICAL',
            'action': 'Immediate freeze. Refer to Anti-Corruption Bureau. Verify with UIDAI Aadhaar database.',
            'recovery': f"₹{amt:,.0f} disbursed; ₹{amt*12:,.0f} projected annual saving",
            'notes': 'Check if multiple bank accounts trace to same phone/address — indicators of a coordinator.'
        }
    if 'LIKELY_DECEASED' in details:
        age = details['LIKELY_DECEASED'].get('age', 90)
        return {
            'fraud_type': 'Deceased Beneficiary Fraud',
            'assessment': f"{name} is {age} years old — no death record removal found. CAG Kerala SSP audit (2023) found 96,285 such cases receiving payments 2–20 months post-death.",
            'severity': 'HIGH',
            'action': 'Verify death certificate with Block Development Officer within 48 hours. Stop disbursements immediately.',
            'recovery': f"₹{amt:,.0f} per cycle; estimated ₹{amt*24:,.0f} overpaid over 2 years",
            'notes': 'Check if nominee/family member is aware — they may be complicit or victims of local agent fraud.'
        }
    if 'SHARED_BANK' in details:
        count = details['SHARED_BANK'].get('count', 3)
        return {
            'fraud_type': 'Shared Bank Fraud Ring',
            'assessment': f"One bank account is collecting payments for {count} different beneficiaries — a single operator extracting multiple welfare funds, matching the pattern in CAG Assam PM-KISAN audit (2024): ₹567 crore fraud.",
            'severity': 'HIGH',
            'action': f'Freeze all {count} accounts linked to this bank number. Initiate field verification for each beneficiary.',
            'recovery': f"₹{amt*count:,.0f} total across all {count} linked accounts",
            'notes': f'Map all {count} accounts on network graph — likely connected to a local village-level intermediary.'
        }
    if 'DUPLICATE_AADHAAR' in details:
        count = details['DUPLICATE_AADHAAR'].get('count', 2)
        return {
            'fraud_type': 'Duplicate Identity Fraud',
            'assessment': f"Aadhaar number used {count} times under different names — direct match to CAG PMJAY finding (2023) where 7.49 lakh ghost beneficiaries were registered, some under a single mobile number (9999999999).",
            'severity': 'HIGH',
            'action': 'Query UIDAI API for legitimate holder. Cancel all duplicate registrations and recover funds.',
            'recovery': f"₹{amt*count:,.0f} total across {count} duplicate entries",
            'notes': 'This Aadhaar number should be blacklisted system-wide after verification.'
        }
    if 'ADDRESS_BOMB' in details:
        count = details['ADDRESS_BOMB'].get('count', 10)
        return {
            'fraud_type': 'Address Concentration Fraud',
            'assessment': f"{count} beneficiaries registered at one address — statistically impossible for a residence. Indicates bulk fabrication by a local middleman inflating scheme coverage numbers.",
            'severity': 'HIGH',
            'action': 'Mandatory physical field visit. Suspend all registrations from this address pending verification.',
            'recovery': f"₹{amt*count:,.0f} if all {count} registrations are fraudulent",
            'notes': 'Check if address is a school, panchayat office, or abandoned property — common bulk-registration sites.'
        }
    if 'AMOUNT_ANOMALY' in details:
        z = details['AMOUNT_ANOMALY'].get('zscore', 3.0)
        return {
            'fraud_type': 'Payment Amount Anomaly',
            'assessment': f"Disbursement of ₹{amt:,.0f} is {z:.1f} standard deviations above scheme average — may indicate scheme misclassification, data entry error, or deliberate amount inflation.",
            'severity': 'MEDIUM',
            'action': 'Review original claim documents. Recalculate entitlement per scheme rules. Recover excess immediately.',
            'recovery': f"Excess over scheme limit; full amount ₹{amt:,.0f} held pending review",
            'notes': 'Compare with neighboring beneficiaries in same scheme/district for baseline reference.'
        }
    return {
        'fraud_type': 'Multiple Irregularities',
        'assessment': f"{name} shows {len(case.get('anomalies',[]))} concurrent anomalies matching documented fraud patterns from multiple CAG audit reports (2023-2025).",
        'severity': case.get('risk_level', 'MEDIUM'),
        'action': 'Full re-verification required. Suspend payments pending Block Development Officer confirmation.',
        'recovery': f"₹{amt:,.0f} pending verification outcome",
        'notes': 'Multiple simultaneous anomalies significantly increase fraud probability over any single flag.'
    }

# ── Detection Engine ──────────────────────────────────────────────
def age_from_dob(s):
    try: return (datetime.today() - datetime.strptime(str(s).strip(), '%Y-%m-%d')).days // 365
    except: return 0

def is_fabricated(acc):
    d = ''.join(c for c in str(acc) if c.isdigit())
    if not d or len(d) < 6: return False
    if len(set(d)) <= 2: return True
    seq = '0123456789'
    return any(seq[i:i+7] in d for i in range(4))

def detect(df):
    df = df.copy()
    df.columns = [c.lower().strip().replace(' ','_') for c in df.columns]
    for col in ['name','aadhaar','bank_account','address','district','scheme','amount','dob','mobile','bank_name','registration_date']:
        if col not in df.columns: df[col] = ''
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)

    am = defaultdict(list); bm = defaultdict(list); adm = defaultdict(list)
    for i, r in df.iterrows():
        am[str(r['aadhaar'])].append(i)
        bm[str(r['bank_account'])].append(i)
        adm[str(r['address'])].append(i)

    vals = df['amount'].values
    mu, sigma = float(np.mean(vals)), float(np.std(vals)) or 1
    out = []

    for i, r in df.iterrows():
        anom, score, det = [], 0, []
        aa = str(r['aadhaar']); ba = str(r['bank_account'])
        ad = str(r['address']); amt = float(r['amount'])
        mob = str(r.get('mobile',''))

        n = len(am[aa])
        if n > 1:
            anom.append(f"Aadhaar linked to {n} different beneficiaries"); score += 40
            det.append({'type':'DUPLICATE_AADHAAR','count':n})

        n = len(bm[ba])
        if n >= 3:
            anom.append(f"Bank account shared with {n} beneficiaries"); score += 35
            det.append({'type':'SHARED_BANK','count':n})

        n = len(adm[ad])
        if n >= 10:
            anom.append(f"Address has {n} registered beneficiaries"); score += 25
            det.append({'type':'ADDRESS_BOMB','count':n})

        if is_fabricated(ba):
            anom.append("Bank account has fabricated/repeated digits"); score += 45
            det.append({'type':'FABRICATED_BANK'})

        age = age_from_dob(r.get('dob',''))
        if age > 85:
            anom.append(f"Beneficiary age {age} yrs — likely deceased"); score += 35
            det.append({'type':'LIKELY_DECEASED','age':age})

        z = (amt - mu) / sigma
        if z > 3:
            anom.append(f"Payment ₹{amt:,.0f} is {z:.1f}σ above average"); score += 20
            det.append({'type':'AMOUNT_ANOMALY','zscore':round(float(z),2)})

        if not mob or mob.strip().lower() in ['nan','none','n/a','0','','null',' ','unknown']:
            anom.append("No valid mobile number — identity unverifiable"); score += 15
            det.append({'type':'MISSING_MOBILE'})

        if anom:
            lvl = 'HIGH' if score >= 55 else 'MEDIUM' if score >= 25 else 'LOW'
            out.append({
                'id': int(i), 'name': str(r['name']), 'aadhaar': aa,
                'bank_account': ba, 'bank_name': str(r['bank_name']),
                'address': ad, 'district': str(r['district']),
                'scheme': str(r['scheme']), 'amount': round(amt, 2),
                'dob': str(r['dob']), 'age': age, 'mobile': mob,
                'registration_date': str(r['registration_date']),
                'anomalies': anom, 'risk_score': min(score, 99),
                'risk_level': lvl, 'risk_details': det,
            })
    return sorted(out, key=lambda x: -x['risk_score'])

def make_graph(cases):
    nodes, edges, seen = [], [], set()
    bg = defaultdict(list); ag = defaultdict(list)
    for c in cases: bg[c['bank_account']].append(c); ag[c['aadhaar']].append(c)

    def N(nid, d):
        if nid not in seen: nodes.append({'data':{'id':nid,**d}}); seen.add(nid)
    def E(s, t, tp):
        eid = f"e_{s}_{t}"
        if eid not in seen: edges.append({'data':{'id':eid,'source':s,'target':t,'etype':tp}}); seen.add(eid)

    for ba, cs in bg.items():
        if len(cs) >= 2:
            h = f"bank_{ba[-8:]}"
            N(h, {'label': f"Bank\n...{ba[-4:]}", 'nodeType':'bank', 'count':len(cs)})
            for c in cs:
                nid = f"p_{c['id']}"
                N(nid, {'label':c['name'][:12],'nodeType':'person','risk':c['risk_level'],'score':c['risk_score'],'caseId':c['id'],'amount':c['amount'],'scheme':c['scheme']})
                E(h, nid, 'shared_bank')

    for aa, cs in ag.items():
        if len(cs) >= 2:
            h = f"aad_{aa[-8:]}"
            N(h, {'label': f"ID\n...{aa[-4:]}", 'nodeType':'aadhaar', 'count':len(cs)})
            for c in cs:
                nid = f"p_{c['id']}"
                N(nid, {'label':c['name'][:12],'nodeType':'person','risk':c['risk_level'],'score':c['risk_score'],'caseId':c['id'],'amount':c['amount'],'scheme':c['scheme']})
                E(h, nid, 'shared_aadhaar')

    return {'nodes': nodes[:200], 'edges': edges[:400]}

# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
def index(): return send_from_directory('frontend', 'index.html')

@app.route('/status')
def status():
    p = get_provider()
    return jsonify({
        'backend': True,
        'provider': p,
        'ai_active': p != 'none',
        'keys': {
            'groq': bool(GROQ_KEY),
            'gemini': bool(GEMINI_KEY),
            'anthropic': bool(ANTHROPIC_KEY)
        }
    })

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file sent'}), 400
        df = pd.read_csv(request.files['file'])
        cases = detect(df)
        graph = make_graph(cases)
        total = len(df); fl = len(cases)
        high  = sum(1 for x in cases if x['risk_level']=='HIGH')
        med   = sum(1 for x in cases if x['risk_level']=='MEDIUM')
        low   = sum(1 for x in cases if x['risk_level']=='LOW')
        leak  = sum(x['amount'] for x in cases)
        bd = defaultdict(int); bs = defaultdict(int)
        bt = {'DUPLICATE_AADHAAR':0,'SHARED_BANK':0,'ADDRESS_BOMB':0,'FABRICATED_BANK':0,'LIKELY_DECEASED':0,'AMOUNT_ANOMALY':0,'MISSING_MOBILE':0}
        for c in cases:
            bd[c['district']] += 1; bs[c['scheme']] += 1
            for d in c['risk_details']:
                if d['type'] in bt: bt[d['type']] += 1
        return jsonify({
            'success': True,
            'mode': 'real',
            'summary': {'total':total,'flagged':fl,'high':high,'medium':med,'low':low,'clean':total-fl,'leakage':round(leak),'rate':round(fl/total*100,1) if total else 0},
            'cases': cases[:500],
            'graph': graph,
            'analytics': {
                'by_district': dict(sorted(bd.items(), key=lambda x:-x[1])[:10]),
                'by_scheme': dict(bs),
                'by_type': bt
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/explain', methods=['POST'])
def explain():
    case = request.json or {}
    provider = get_provider()
    smart = smart_explanation(case)

    if provider == 'none':
        smart['ai_enabled'] = False
        smart['provider'] = 'none'
        return jsonify(smart)

    prompt = f"""You are a senior CAG India fraud auditor. Analyze this welfare fraud case and respond in EXACTLY this JSON format with no other text:

Case: {case.get('name')}, Age {case.get('age')}, {case.get('scheme')}, {case.get('district')}
Amount: ₹{case.get('amount',0):,.0f}  Risk Score: {case.get('risk_score')}/100
Anomalies: {'; '.join(case.get('anomalies', []))}

Respond ONLY with valid JSON:
{{
  "fraud_type": "3-5 word pattern name",
  "assessment": "2 sentence explanation referencing real CAG findings",
  "severity": "CRITICAL or HIGH or MEDIUM",
  "action": "one specific instruction for field officer",
  "recovery": "estimated recovery amount with justification",
  "notes": "one unique investigation step for this specific case"
}}"""

    try:
        text, used_provider = call_ai(prompt)
        # Parse JSON response
        text = text.strip()
        start = text.find('{'); end = text.rfind('}') + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            result['ai_enabled'] = True
            result['provider'] = used_provider
            return jsonify(result)
        else:
            smart['ai_enabled'] = True
            smart['provider'] = used_provider
            smart['raw'] = text
            return jsonify(smart)
    except Exception as e:
        print(f"[AI Error] {e}")
        smart['ai_enabled'] = False
        smart['provider'] = 'error'
        smart['error'] = str(e)
        return jsonify(smart)

if __name__ == '__main__':
    p = get_provider()
    print("\n" + "="*50)
    print("  GhostAudit AI — Starting")
    print(f"  Groq    : {'✓ ACTIVE' if GROQ_KEY else '✗ not set'}")
    print(f"  Gemini  : {'✓ ACTIVE' if GEMINI_KEY else '✗ not set'}")
    print(f"  Claude  : {'✓ ACTIVE' if ANTHROPIC_KEY else '✗ not set'}")
    print(f"  AI Mode : {'✓ ' + p.upper() if p != 'none' else '✗ No key found — edit .env file'}")
    print(f"  URL     : http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
