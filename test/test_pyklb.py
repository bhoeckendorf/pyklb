import os, urllib
import numpy as np
import pyklb
import unittest


class KlbTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.testread_filepath = os.path.join(os.getcwd(), "deleteme.klb")
        self.testwrite_filepath = self.testread_filepath.replace("deleteme.klb", "deleteme2.klb")

        # version (by commit id) of main library to use
        klbCommitId = "5edcaecc858911c7b3855579bde5cb3116cb4680"
        # download required KLB test image
        klbUrl = "https://bitbucket.org/fernandoamat/keller-lab-block-filetype/raw/%s" % klbCommitId

        print("Downloading test image to %s ..." % self.testread_filepath)
        if os.path.exists(self.testread_filepath):
            raise Error("%s exists, please move or delete the file" % self.testread_filepath)
        urllib.urlretrieve( "%s/testData/img.klb" % klbUrl, self.testread_filepath )
        if not os.path.exists(self.testread_filepath):
            raise Error("Downloading failed, tests aborted")

    @classmethod
    def tearDownClass(self):
        for fp in [self.testread_filepath, self.testwrite_filepath]:
            if not os.path.exists(fp):
                continue
            print("Removing temporary file %s" % fp)
            os.remove(fp)
            if os.path.exists(fp):
                raise Error("Failed to remove temporary file %s,\n    please delete it manually" % fp)

    def test_header(self):
        header = pyklb.readheader(self.testread_filepath)
        self.assertEqual(np.prod(header["imagesize_tczyx"]), 442279)

    def test_readfull(self):
        img = pyklb.readfull(self.testread_filepath)
        self.assertEqual(np.prod(img.shape), 442279)
        self.assertEqual(round(np.mean(img)), 352)

    def test_readroi(self):
        lb = [9, 15, 15]
        ub = [11, 99, 99]
        img = pyklb.readroi(self.testread_filepath, lb, ub)
        self.assertEqual(np.prod(img.shape), np.prod([ 1+ub[i]-lb[i] for i in range(len(lb)) ]))
        self.assertEqual(round(np.mean(img)), 568)

        with self.assertRaises(IndexError):
            img = pyklb.readroi(self.testread_filepath, [9, 15, 15], [66, 99, 99])
            img = pyklb.readroi(self.testread_filepath, [-1, 15, 15], [11, 99, 99])

    def test_readfull_inplace(self):
        header = pyklb.readheader(self.testread_filepath)

        img = np.zeros(header["imagesize_tczyx"], header["datatype"])
        pyklb.readfull_inplace(img, self.testread_filepath, nochecks=False)
        self.assertEqual(round(np.mean(img)), 352)

        img = np.zeros(header["imagesize_tczyx"], header["datatype"])
        pyklb.readfull_inplace(img, self.testread_filepath, nochecks=True)
        self.assertEqual(round(np.mean(img)), 352)

        img = np.zeros(header["imagesize_tczyx"], np.uint8)
        with self.assertRaises(TypeError):
            pyklb.readfull_inplace(img, self.testread_filepath)

        img = np.zeros([666, 666], header["datatype"])
        with self.assertRaises(IOError):
            pyklb.readfull_inplace(img, self.testread_filepath)

    def test_readroi_inplace(self):
        header = pyklb.readheader(self.testread_filepath)
        lb = [9, 15, 15]
        ub = [11, 99, 99]

        img = np.zeros(1 + np.array(ub) - np.array(lb), header["datatype"])
        pyklb.readroi_inplace(img, self.testread_filepath, lb, ub, nochecks=False)
        self.assertEqual(round(np.mean(img)), 568)

        img = np.zeros(1 + np.array(ub) - np.array(lb), header["datatype"])
        pyklb.readroi_inplace(img, self.testread_filepath, lb, ub, nochecks=True)
        self.assertEqual(round(np.mean(img)), 568)

        img = np.zeros(1 + np.array(ub) - np.array(lb), np.uint8)
        with self.assertRaises(TypeError):
            pyklb.readroi_inplace(img, self.testread_filepath, lb, ub)

        img = np.zeros([666, 666], header["datatype"])
        with self.assertRaises(IOError):
            pyklb.readroi_inplace(img, self.testread_filepath, lb, ub)

        img = np.zeros(1 + np.array(ub) - np.array(lb), header["datatype"])
        with self.assertRaises(IOError):
            pyklb.readroi_inplace(img, self.testread_filepath, ub, lb)

    def test_writefull(self):
        print("Testing KLB writing to %s" % self.testwrite_filepath)
        if os.path.exists(self.testwrite_filepath):
            print("Skipping writing tests because file %s exists.\n    Please move or delete this file and re-run the tests." % self.testwrite_filepath)
            return

        img = pyklb.readfull(self.testread_filepath)
        pyklb.writefull(img, self.testwrite_filepath, pixelspacing_tczyx=[0.5, 0.5, 5.0])
        self.assertTrue(os.path.exists(self.testwrite_filepath))
        img2 = pyklb.readfull(self.testwrite_filepath)
        self.assertEquals(img.dtype, img2.dtype)
        self.assertEquals(img.shape, img2.shape)
        self.assertEquals(np.mean(img), np.mean(img2))

        header = pyklb.readheader(self.testwrite_filepath)
        self.assertTrue(np.all( header["pixelspacing_tczyx"] == [1, 1, 0.5, 0.5, 5.0] ))


if __name__ == '__main__':
    unittest.main()
