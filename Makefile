upload:
	rsync -av --exclude-from=".gitignore" ./ AiMax:/opt/data/private/FollmerFlow/

download :
	rsync -av --exclude="*.pth" AiMax:/opt/data/private/FollmerFlow/assets ./

clean:
	@if [ -d "assets" ]; then \
		rm -rf "assets"; \
	fi
	
cleanpdf:
	rm -rf assets/*.pdf