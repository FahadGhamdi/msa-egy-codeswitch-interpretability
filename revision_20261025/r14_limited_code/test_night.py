"""CPU tests for planned resume, corruption/identity guards, design and statistics."""
import unittest,tempfile,json,sqlite3,hashlib
from pathlib import Path
import numpy as np
from checkpoint import Store
from run_model import check_bridge,ARRAYS,order_items
from run_queue import next_action
from analyze import summarize
P=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_resume_no_duplicate(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'x.sqlite';d=Store(p,{'a':1});d.set('status','RUNNING');d.put(0,1,{'x':np.arange(3)}, {'passed':True},np);d.set('status','PAUSED_BUDGET');d.close()
   d=Store(p,{'a':1},resume=True);self.assertTrue(d.has(0,1));self.assertEqual(d.count(),1)
   with self.assertRaises(RuntimeError):d.put(0,1,{'x':np.zeros(3)}, {},np)
   d.close()
 def test_identity_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'x.sqlite';d=Store(p,{'a':1});d.close()
   with self.assertRaises(RuntimeError):Store(p,{'a':2},True)
 def test_crash_and_failure_not_resumed(self):
  for status in ['RUNNING','STOPPED_PENDING_REVIEW']:
   with tempfile.TemporaryDirectory() as t:
    p=Path(t)/'x.sqlite';d=Store(p,{});d.set('status',status);d.close()
    with self.assertRaises(RuntimeError):Store(p,{},True)
 def test_corruption_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'x.sqlite';d=Store(p,{});d.put(0,0,{'x':np.arange(3)}, {},np);d.set('status','PAUSED_BUDGET');d.close()
   con=sqlite3.connect(p);con.execute("UPDATE batches SET payload=?",(b'bad',));con.commit();con.close()
   with self.assertRaises(RuntimeError):Store(p,{},True)
 def test_queue_policy(self):
  self.assertEqual(next_action(75,'PAUSED_BUDGET'),'resume');self.assertEqual(next_action(0,'COMPLETE'),'complete')
  for code,status in [(1,'PAUSED_BUDGET'),(75,'RUNNING'),(0,'STOPPED_PENDING_REVIEW'),(-9,'RUNNING')]:self.assertEqual(next_action(code,status),'stop')
 def test_bridge_failure(self):
  x={k:np.ones((2,4),np.float32) for k in ARRAYS};y={k:v.copy() for k,v in x.items()};self.assertTrue(check_bridge(x,y,np)[0]);y['metrics'][0,0]=-1;self.assertFalse(check_bridge(x,y,np)[0])
 def test_bootstrap_paired_and_item_weighted(self):
  x=np.array([[1.,3.,-2.],[1.,3.,-2.],[7.,9.,4.]])
  p,ci,n,C,e=summarize(x,[('a',),('a',),('b',)],19)
  np.testing.assert_allclose(p,[3,5,0]);np.testing.assert_allclose(ci[:,1]-ci[:,0],2);np.testing.assert_allclose(ci[:,2]-ci[:,0],-3);self.assertEqual((n,C,e),(3,2,0))
 def test_missingness(self):
  self.assertIsNone(summarize(np.full((2,3),np.nan),[('a',),('b',)]))
  with self.assertRaises(RuntimeError):summarize(np.array([[1,np.nan],[2,3]]),[('a',),('b',)])
 def test_draw_multiplicity(self):
  # 99 identical draws must keep 99% weight; unique-set mean would be wrong.
  logical=np.array([0]*99+[1]);values=np.array([1.,101.]);self.assertEqual(values[logical].mean(),2.)
 def test_frozen_limited_design(self):
  import collections
  for m in (P/'models').iterdir():
   d=json.loads((m/'LIMITED_DESIGN.json').read_text());old=json.loads((m/'DESIGN.json').read_text());rank=old['heads'][20]
   self.assertEqual(d['active_k'],[1,10,20]);active=[d['batches'][b] for b in d['active_batches']];ids=[c for b in active for c in b['condition_ids']]
   self.assertEqual(len(ids),len(set(ids)))
   with np.load(m/'reference_means_norms.npz') as z:
    for draw,heads in enumerate(d['draw_sequences']):
     self.assertEqual(len(set(map(tuple,heads))),20)
     for k in [1,10,20]:
      cid=d['logical_to_unique'][f'{k}:{draw}'];self.assertIn(cid,ids);self.assertEqual(set(map(tuple,d['conditions'][cid]['heads'])),set(map(tuple,heads[:k])))
      self.assertEqual(collections.Counter(x[0] for x in heads[:k]),collections.Counter(x[0] for x in rank[:k]))
     for s,t in zip(rank,heads):
      ratio=z[f'norm_{s[0]}'][t[1]]/max(z[f'norm_{s[0]}'][s[1]],1e-8);self.assertEqual(s[0],t[0]);self.assertTrue(.5<=ratio<=2)
if __name__=='__main__':unittest.main()
