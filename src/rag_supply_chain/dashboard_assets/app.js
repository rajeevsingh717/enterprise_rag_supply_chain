const $ = (id) => document.getElementById(id);
const pct = (value) => `${(value * 100).toFixed(value >= .1 ? 0 : 1)}%`;
const scoreLabels = { retrieval_hit: "Retrieval hit", citation_hit: "Citation hit", faithfulness: "Faithfulness", answer_relevance: "Answer relevance", context_precision: "Context precision", reciprocal_rank: "Mean reciprocal rank" };

fetch('/api/dashboard').then(r => r.json()).then(data => {
  if (data.error) throw new Error(data.error);
  const allOk = data.services.every(s => s.ok);
  $('overall').textContent = allOk ? 'Local stack healthy' : 'Sample mode · some services offline';
  document.querySelector('.pulse').classList.toggle('ok', allOk);
  $('services').innerHTML = data.services.map(s => `<div class="service ${s.ok ? 'ok' : ''}"><strong>${s.name}</strong><small>${s.ok ? s.detail : 'Unavailable'}</small></div>`).join('');
  $('pipeline').innerHTML = data.pipeline.map((s,i) => `<div class="stage"><i>${String(i+1).padStart(2,'0')}</i><div><strong>${s.name}</strong><p>${s.detail}</p></div></div>`).join('');
  $('doc-title').textContent = data.document.title;
  $('doc-meta').textContent = `${data.document.name} · ${data.document.sections} sections · safe checked-in corpus`;
  $('topics').innerHTML = data.document.topics.map(t => `<span>${t}</span>`).join('');
  $('query-label').textContent = data.sample_query.label;
  $('question').textContent = data.sample_query.question;
  $('answer').textContent = data.sample_query.answer;
  $('citation').textContent = data.sample_query.citation;
  $('case-count').textContent = `${data.evaluation.case_count} QA cases`;
  const keys = ['retrieval_hit','citation_hit','faithfulness','answer_relevance','context_precision','reciprocal_rank'];
  $('scores').innerHTML = keys.map(k => `<div class="score"><strong>${pct(data.evaluation.metrics[k])}</strong><span>${scoreLabels[k]}</span></div>`).join('');
  const n = data.normalization;
  $('before').textContent = n.before_tokens.toLocaleString(); $('after').textContent = n.after_tokens.toLocaleString();
  $('reduction').textContent = pct(n.reduction_fraction); $('ring').style.setProperty('--pct', `${n.reduction_fraction*360}deg`);
  $('target').textContent = n.target_met ? '15% target met' : '15% target not met';
}).catch(err => { $('overall').textContent = `Dashboard data unavailable: ${err.message}`; });
