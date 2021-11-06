$Folder1 = Get-childitem '../data/PennFudanPed/Test/images'
$Folder2 = Get-childitem '../data/PennFudanPed/Test/masks'
Compare-Object $Folder1 $Folder2 -Property BaseName