<?php
if( isset($_POST['firstname'] ) && isset( $_POST['lastname'] ) )
{
    echo "Storing data"
    $txt= $_POST['firstname'].' - '.$_POST['lastname'] . PHP_EOL; 
    file_put_contents('data.txt', $txt, FILE_APPEND);
}
?>