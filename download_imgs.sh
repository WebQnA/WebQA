#!/usr/bin/bash

end=10
x=1


until [ $x -eq $end ]; do

    wget http://tiger.lti.cs.cmu.edu/yingshac/WebQA_data_first_release/WebQA_imgs_7z_chunks/imgs.7z.00$x
    x=$(($x+1))
        
done
 
end=52
until [ $x -eq $end ]; do
    wget http://tiger.lti.cs.cmu.edu/yingshac/WebQA_data_first_release/WebQA_imgs_7z_chunks/imgs.7z.0$x
    x=$(($x+1))
done

wget http://tiger.lti.cs.cmu.edu/yingshac/WebQA_data_first_release/imgs.lineidx
