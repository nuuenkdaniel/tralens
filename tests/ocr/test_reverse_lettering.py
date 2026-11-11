import unittest

from PIL import Image,ImageDraw

from .test_automatic_units import AutomaticUnitTests,box


class ReverseLettering(unittest.TestCase):
    def test_white_columns_form_one_run_but_real_light_divider_blocks(self):
        fixture=AutomaticUnitTests();fixture.setUp()
        image=Image.new('RGB',(160,140),'white');draw=ImageDraw.Draw(image)
        draw.rectangle((10,10,140,125),fill='black')
        boxes=[]
        for i,x in enumerate((45,70,95)):
            boxes.append(box(i,(x-2,30,x+12,103),'かなかな'))
            for y in (35,55,75,95):
                draw.line((x,y,x+8,y,x+8,y+5,x+2,y+5),fill='white',width=2)
        together=fixture.propose(boxes,image)
        self.assertEqual([p.main_ids for p in together.proposals],[(2,1,0)])
        fixture.assert_partition(together)
        draw.line((89,12,89,123),fill='white',width=3)
        divided=fixture.propose(boxes,image)
        self.assertGreater(len(divided.proposals),1)
        fixture.assert_partition(divided)


if __name__=='__main__':unittest.main()
