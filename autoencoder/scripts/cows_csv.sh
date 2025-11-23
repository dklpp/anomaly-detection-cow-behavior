#!/bin/bash

echo -e ""
echo "All dates:"
cat $1 | grep -E '(09-30|10-01|10-02)' | awk -F ',' '{print $1" "$3}'
videos=$(ls ../cow_videos/flipped/)

echo -e ""

echo "Videos:"
while read line; do
	date=$(echo $line | cut -d ' ' -f 1 | sed 's/-//g')
	time=$(echo $line | cut -d ' ' -f 2)
	hour=$(echo $line | cut -d ' ' -f 2 | cut -d ':' -f 1)
	minute=$(echo $line | cut -d ' ' -f 2 | cut -d ':' -f 2)
	hourMinusOne=$(echo ${hour#0}); hourMinusOne=$((hourMinusOne - 1));
	[ $hourMinusOne -lt 10 ] && hourMinusOne="0${hourMinusOne}"
	duration=$(echo $line | cut -d ' ' -f 3)
	time=$(echo $time | awk -F ':' '{print $1$2$3}')
	file="${date}_${hour}"
	file2="${date}-${hourMinusOne}"

	for videoRoi in $videos; do
		video=$(echo $videoRoi | awk -F '_' '{print $2"_"$3}')
		videoMin=$(echo $video | cut -d '_' -f 2 | cut -d '.' -f 1)
		videoMin=${videoMin:2:2}
		if [[ $video = $file* ]] || [[ $video = $file2* ]]; then
			if [ ${videoMin#0} -lt ${minute#0} ]; then
				time=$(echo $time | sed 's/^0*//')
				videoTime=$(echo $video | cut -d '_' -f 2 | cut -d '.' -f 1 | sed 's/^0*//')
				time=$(echo $((time - videoTime)) | sed 's/\([0-9]\{2\}\)\([0-9]\{2\}\)/00:\1:\2/')
				echo "$video $time $duration"
			fi
		fi
	done
done < <(cat $1 | grep -E '(09-30|10-01|10-02)' | awk -F ',' '{print $1" "$3}')
