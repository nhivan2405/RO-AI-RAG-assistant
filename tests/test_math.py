import unittest
from src.pipeline import cosine_manual,make_chunks
class MathTest(unittest.TestCase):
    def test_cosine_equal(self):self.assertAlmostEqual(cosine_manual([1,2],[2,4]),1.0)
    def test_cosine_orthogonal(self):self.assertAlmostEqual(cosine_manual([1,0],[0,1]),0.0)
    def test_cosine_opposite(self):self.assertAlmostEqual(cosine_manual([1,0],[-1,0]),-1.0)
    def test_chunks(self):
        chunks=make_chunks([("test",1,"a "*200)],chunk_size=80,chunk_overlap=10)
        self.assertTrue(len(chunks)>1)
if __name__=="__main__":unittest.main()
