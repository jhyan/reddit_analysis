import unittest
import util

class TestAlpha(unittest.TestCase):

    def test_alpha(self):
        self.assertEqual(util.alpha_filter("Jimmy"),True)
        self.assertEqual(util.alpha_filter("hi JIMMY"),False)
        self.assertEqual(util.alpha_filter("Jim!!!!!my"),True)
        self.assertEqual(util.alpha_filter("Jim12my"),False)
        self.assertEqual(util.alpha_filter("my_name_is_Jimmy"),True)

if __name__ == "__main__":
    unittest.main()


