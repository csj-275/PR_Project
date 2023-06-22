
fid=fopen('rawdata/1228'); 
I = fread(fid);imagesc(reshape(I, 128, 128)'); 
colormap(gray(256));

