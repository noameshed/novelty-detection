<?php
 $path = 'data.txt';
 if (isset($_POST['fname']) && isset($_POST['lname'])) {
    $fh = fopen($path,"a+");
    $string = $_POST['fname'].' - '.$_POST['lname'];
    fwrite($fh,$string); // Write information to the file
    fclose($fh); // Close the file
 }
?>