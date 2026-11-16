import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp

st.set_page_config(page_title='Transaction Monitor', layout='wide')

# The five raw inputs the model needs, with plain-English names for messages.
INPUTS = {
    'amount': 'Amount ($)',
    'customer_typical_amount': "Customer's typical amount ($)",
    'transactions_last_hour': 'Transactions in the last hour',
    'minutes_since_previous': 'Minutes since previous transaction',
    'account_age_days': 'Account age (days)',
}


DRIFT_WARMUP = 100      
DRIFT_WINDOW = 200      
DRIFT_P = 1e-4          
DRIFT_FEATURES = {
    'log_amount': 'amount',
    'amount_ratio': 'amount compared with the customer\'s typical amount',
    'log_gap': 'minutes since previous transaction',
    'log_age': 'account age',
}

# Load trained model
try:
    bundle = joblib.load('model.joblib')
    model = bundle['model']
    history_scores = bundle['history_scores']
    features = bundle['features']
except Exception as e:  # missing file, wrong library version, damaged file
    st.error('**The model could not be loaded, so no transactions can be scored.**\n\n'
             f'Technical detail: `{type(e).__name__}: {e}`\n\n'
             'What to do: make sure `model.joblib` is in the app folder and that the '
             'packages in `requirements.txt` are installed. Until this is fixed, send '
             'transactions through the manual review process.')
    st.stop()

try:
    history_raw = pd.read_csv('history.csv')
    ranges = {c: (history_raw[c].min(), history_raw[c].max()) for c in INPUTS}
    safety_checks_on = True
except Exception:
    history_raw, ranges, safety_checks_on = None, {}, False


def make_features(amount, typical, last_hour, gap, age):
    """The five model features. Works on single numbers or on whole columns."""
    return pd.DataFrame({
        'log_amount': np.log1p(amount),
        'amount_ratio': amount / typical,
        'transactions_last_hour': last_hour,
        'log_gap': np.log1p(gap),
        'log_age': np.log1p(age),
    }, index=[0] if np.isscalar(amount) else None)[features]


def score(amount, typical, last_hour, gap, age):
    """Turn one transaction into the five features and return its anomaly score."""
    row = make_features(amount, typical, last_hour, gap, age)
    return -model.score_samples(row)[0]


history_features = (make_features(history_raw['amount'], history_raw['customer_typical_amount'],
                                  history_raw['transactions_last_hour'],
                                  history_raw['minutes_since_previous'],
                                  history_raw['account_age_days'])
                    if safety_checks_on else None)


# ---- Protection 1: check each transaction's data before scoring it ----
def check_inputs(values):
    """Return (errors, out_of_range).

    errors: the data is missing or impossible. The transaction must NOT be scored,
        because the model silently turns such values into a normal-looking score.
    out_of_range: the value is possible but outside everything the model was built
        on. The score is not reliable there, so the transaction is never cleared
        automatically.
    """
    errors, out_of_range = [], []
    for col, label in INPUTS.items():
        v = values.get(col)
        if v is None or pd.isna(v):
            errors.append(f'{label} is missing or not a number')
    if errors:
        return errors, out_of_range

    if values['amount'] <= 0:
        errors.append('Amount must be greater than $0')
    if values['customer_typical_amount'] <= 0:
        errors.append("Customer's typical amount must be greater than $0")
    for col in ['transactions_last_hour', 'account_age_days']:
        if values[col] < 0 or values[col] != int(values[col]):
            errors.append(f'{INPUTS[col]} must be a whole number of 0 or more')
    if values['minutes_since_previous'] < 0:
        errors.append('Minutes since previous transaction cannot be negative')
    if errors:
        return errors, out_of_range

    for col, (lo, hi) in ranges.items():
        if not lo <= values[col] <= hi:
            out_of_range.append(f'{INPUTS[col]} = {values[col]:,g} '
                                f'(past transactions ranged from {lo:,g} to {hi:,g})')
    return errors, out_of_range


# ---- Protection 2: check whether the stream still looks like history ----
def stream_shift(recent_features):
    """Compare recent transactions with history. Return a description of the
    first input that has clearly shifted, or None if everything looks normal."""
    for f, label in DRIFT_FEATURES.items():
        p = ks_2samp(recent_features[f], history_features[f]).pvalue
        if p < DRIFT_P:
            hist_med = history_features[f].median()
            new_med = recent_features[f].iloc[-25:].median()  # latest values, for the message
            if f.startswith('log_'):  # show typical values in the original units
                hist_med, new_med = np.expm1(hist_med), np.expm1(new_med)
            return label, hist_med, new_med
    return None


# ---- sidebar: the review budget ----
st.title('Transaction Monitor')
st.write('Each new transaction is scored by Isolation Forest. '
         'If the score is above the cutoff, it goes to review.')

pct = st.sidebar.slider('Send this share of transactions to review (%)', 1, 10, 4)
cutoff = np.percentile(history_scores, 100 - pct)
st.sidebar.write(f'Score cutoff: **{cutoff:.3f}**')
st.sidebar.caption('The cutoff is set from past transactions: '
                   'a transaction scoring above it is more unusual than the top '
                   f'{pct}% of history.')
if not safety_checks_on:
    st.sidebar.warning('`history.csv` was not found, so range checks and stream checks are '
                       'OFF. Scores are still shown, but a transaction outside normal ranges '
                       'or a broken data feed will not be caught. Restore `history.csv` '
                       'to turn the checks back on.')

tab1, tab2 = st.tabs(['Check one transaction', 'Live stream'])

# ---- tab 1: type in one transaction ----
with tab1:
    c1, c2 = st.columns(2)
    amount = c1.number_input('Amount ($)', min_value=0.01, value=60.0)
    typical = c1.number_input("Customer's typical amount ($)", min_value=0.01, value=50.0)
    last_hour = c1.number_input('Transactions in the last hour', min_value=0, value=1)
    gap = c2.number_input('Minutes since previous transaction', min_value=0.0, value=120.0)
    age = c2.number_input('Account age (days)', min_value=0, value=700)

    if st.button('Score this transaction'):
        values = dict(amount=amount, customer_typical_amount=typical,
                      transactions_last_hour=last_hour, minutes_since_previous=gap,
                      account_age_days=age)
        errors, out_of_range = check_inputs(values)
        if errors:
            st.error('**NOT SCORED: the data is not valid.**\n\n- ' + '\n- '.join(errors) +
                     '\n\nCorrect the values and score again. If they came from a system '
                     'this way, treat the transaction as needing review.')
        else:
            s = score(amount, typical, last_hour, gap, age)
            if out_of_range:
                st.warning(f'**REVIEW: outside the model\'s experience** (score {s:.3f}, '
                           f'cutoff {cutoff:.3f})\n\n- ' + '\n- '.join(out_of_range) +
                           '\n\nThe model has never seen values like this, so its score '
                           'cannot be trusted to clear the transaction. Check that the '
                           'values are entered correctly; if they are, have a reviewer '
                           'look at it.')
            elif s > cutoff:
                st.error(f'REVIEW  (score {s:.3f}, cutoff {cutoff:.3f})')
            else:
                st.success(f'OK  (score {s:.3f}, cutoff {cutoff:.3f})')
            share = (history_scores < s).mean() * 100
            st.write(f'This is more unusual than {share:.0f}% of past transactions.')

# ---- tab 2: transactions arriving one by one ----
with tab2:
    st.write('Press the button. The app receives the new transactions one at a time '
             'and scores each one as it arrives.')
    speed = st.select_slider('Speed', options=['Slow', 'Medium', 'Fast'], value='Medium')
    pause = {'Slow': 0.5, 'Medium': 0.15, 'Fast': 0.02}[speed]

    if st.button('Start the stream'):
        try:
            stream = pd.read_csv('new_transactions.csv')
        except Exception as e:
            st.error(f'**The transaction file could not be read** (`{type(e).__name__}: {e}`). '
                     'No transactions were scored. Check that `new_transactions.csv` exists '
                     'and is a valid CSV, then start the stream again.')
            st.stop()

        missing_cols = [c for c in ['transaction_id', *INPUTS] if c not in stream.columns]
        if missing_cols:
            st.error('**The stream was not started: the file is missing required columns:** '
                     + ', '.join(f'`{c}`' for c in missing_cols) +
                     '. No transactions were scored. Ask the data team to fix the export; '
                     'the file must contain ' + ', '.join(f'`{c}`' for c in INPUTS) + '.')
            st.stop()

        # Turn text like "N/A" into missing values so Protection 1 can catch it.
        for c in INPUTS:
            stream[c] = pd.to_numeric(stream[c], errors='coerce')

        box = st.empty()
        counts = st.empty()
        alarm = st.empty()
        st.subheader('Sent to review')
        table = st.empty()
        st.subheader('Held: data problem (not scored)')
        held_table = st.empty()
        flagged, held = [], []
        recent = []          # features of recently scored transactions, for Protection 2
        n_ok = 0
        stopped_at = None

        for i, r in stream.iterrows():
            values = {c: r[c] for c in INPUTS}
            errors, out_of_range = check_inputs(values)

            if errors:
                held.append({'id': r['transaction_id'], 'problem': '; '.join(errors)})
                box.warning(f"{r['transaction_id']}  ->  HELD: data problem ({errors[0]})")
            else:
                s = score(*values.values())
                recent.append(make_features(*values.values()).iloc[0])
                if s > cutoff or out_of_range:
                    flagged.append({'id': r['transaction_id'], 'amount': r['amount'],
                                    'typical': r['customer_typical_amount'],
                                    'last_hour': r['transactions_last_hour'],
                                    'score': round(s, 3),
                                    'reason': ('outside model range: ' + '; '.join(out_of_range))
                                              if out_of_range else 'high score'})
                    box.error(f"{r['transaction_id']}: ${r['amount']:.2f}  ->  REVIEW (score {s:.3f})")
                else:
                    n_ok += 1
                    box.info(f"{r['transaction_id']}: ${r['amount']:.2f}  ->  ok (score {s:.3f})")

            counts.write(f'Received **{i + 1}** of {len(stream)}.  '
                         f'Sent to review: **{len(flagged)}**.  OK: **{n_ok}**.  '
                         f'Held for data problems: **{len(held)}**')
            if flagged:
                table.dataframe(pd.DataFrame(flagged), hide_index=True)
            if held:
                held_table.dataframe(pd.DataFrame(held), hide_index=True)

            # Protection 2: once there is enough data, check the recent stream.
            if safety_checks_on and len(recent) >= DRIFT_WARMUP:
                shift = stream_shift(pd.DataFrame(recent[-DRIFT_WINDOW:]))
                if shift:
                    label, hist_med, new_med = shift
                    stopped_at = i
                    window_start = stream['transaction_id'].iloc[max(0, i + 1 - DRIFT_WINDOW)]
                    alarm.error(
                        f'**STREAM STOPPED after {r["transaction_id"]}: the incoming data no '
                        f'longer looks like the data the model was built on.**\n\n'
                        f'Input that shifted: **{label}**. Typical value in past '
                        f'transactions: **{hist_med:,.2f}**. Typical value in the latest '
                        f'transactions: **{new_med:,.2f}**. A sudden shift like this is '
                        'often a feed problem (for example, a field sent in different units) '
                        'rather than a real change in customers. Either way, the scores and '
                        'the review cutoff are no longer reliable.\n\n'
                        f'**What to do:**\n'
                        f'1. The {len(flagged)} transactions already sent to review still '
                        'need review.\n'
                        f'2. Do **not** treat the "ok" results from {window_start} onward '
                        'as cleared. They were scored on shifted data.\n'
                        f'3. The remaining {len(stream) - i - 1} transactions were not '
                        'scored and are on hold.\n'
                        '4. Ask the data team to check the feed for this field. Restart the '
                        'stream once it is fixed.')
                    break
            time.sleep(pause)

        # at the end, look at the answer key
        if stopped_at is None:
            if 'needs_review' in stream.columns:
                answer = stream.set_index('transaction_id')['needs_review']
                hits = sum(answer[f['id']] for f in flagged)
                st.success(f'Done. {len(flagged)} transactions were sent to review, '
                           f'and {hits} of them really needed review. '
                           f'In total, {answer.sum()} transactions needed review.')
            else:
                st.success(f'Done. {len(flagged)} transactions were sent to review.')
            if held:
                st.warning(f'{len(held)} transactions were held because their data was '
                           'missing or impossible. They were not scored. Fix the source data '
                           'and re-run them, or have a reviewer check them by hand.')
