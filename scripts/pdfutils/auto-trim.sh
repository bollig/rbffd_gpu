#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

echo "Auto-trimming image..."
convert  -density 250 -trim +repage $1 tmp_out.pdf
convert  -density 250 -trim +repage tmp_out.pdf compressed_$1

echo "Compressing trimmed image..."
cat > "COMPRESS_PDF.qfilter" <<STOP
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Domains</key>
	<dict>
		<key>Applications</key>
		<true/>
		<key>Printing</key>
		<true/>
	</dict>
	<key>FilterData</key>
	<dict>
		<key>ColorSettings</key>
		<dict>
			<key>ImageSettings</key>
			<dict>
				<key>ImageScaleSettings</key>
				<dict>
					<key>ImageResolution</key>
					<integer>200</integer>
					<key>ImageScaleFactor</key>
					<real>0.0</real>
					<key>ImageScaleInterpolate</key>
					<integer>3</integer>
					<key>ImageSizeMax</key>
					<integer>0</integer>
					<key>ImageSizeMin</key>
					<integer>0</integer>
				</dict>
			</dict>
		</dict>
	</dict>
	<key>FilterType</key>
	<integer>1</integer>
	<key>Name</key>
	<string>Reduce File Size (Scale 75%)</string>
</dict>
</plist>
STOP

sleep 1 
# Now use the filter. 
# ONLY WORKS ON OSX, BUT IT WORKS REALLY REALLY WELL. 
automator -v -i compressed_$1 $SCRIPT_DIR/CompressPDF.workflow

sleep 1 
chmod 644 compressed_$1
# Cleanup: 
rm COMPRESS_PDF.qfilter
rm tmp_out.pdf
