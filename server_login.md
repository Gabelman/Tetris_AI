# Log into lsv server
The simple and straight-forward way to log into the server is to first go on the contact.lsv.uni-saarland.server. In a terminal type:
```console
florian@fedora$ssh {Username}@contact.lsv.uni-saarland.de
```
To then go on the server on which jobs can be submitted:
```console
florian@fedora$ssh {Username}@submit.lsv.uni-saarland.de
```
Here, replace `{Username}` with your respective username.\
**Note**: In the process you will be prompted for your password.
# Create connection through ssh-keys
This is really convenient, as it allows to login without having to type in yout passwords all the time. Also this is important to create a connection through VSCode later on.\
To set-up ssh keys, run:
```console
florian@fedora$ssh-keygen
```
in a terminal (on Linux). This will create the files `id_rsa` and `id_rsa.pub` in `home/{Username}/.ssh/`.\
Next, the public key needs to be added to the remote server. To do so, simply use the command:
```console
florian@fedora$ssh-copy-id {Username}@contact.lsv.uni-saarland.de
```
Here you will be prompted one last time to input your password. After that try and test whether you have access just with:
```console
florian@fedora$ssh {Username}@contact.lsv.uni-saarland.de
```
(Exit again just by typing `exit`)
### Connect through jump-host
In order to have direct access to the _submit.lsv.uni-saarland.de_ server, the key has to be added to the `authorized_keys` of that server. To do that, the local public key must be appended to the file `authorized_keys` in _{Username}@submit.lsv.uni-saarland.de:~/.ssh/authorized_keys_. This can be done with the command
```console
florian@fedora$ssh -J {Username}@contact.lsv.uni-saarland.de {Username}@submit.lsv.uni-saarland.de "cat >> ~/.ssh/authorized_keys" < ~/.ssh/id_rsa.pub
```
This executes the command `cat >> ~/.ssh/authorized_keys`, on the submit.lsv.uni-saarland.de server. To break it down:
- `cat` "concatenates" the contents of files, in this case just one file. Hence, it just gives the content of the file.
- `>>` puts the content in another file. The two arrows mean that the content is appended and doesn't overwrite the file (in contrast to `>`, which would overwrite).
- `<` pipes the local file into the command. This means that the `cat` of the command will output the content of the local id_rsa.pub file and then append it to `authorized_keys`.
- The -J flag of the ssh command specifies a jump host. This is why we give the two remote locations as arguments.

# Setting up SSH config
To run VSCode on the remote server, as well as for even easier ssh access, we can set up a config file in our `.ssh` folder. To do so, we can simply create the file
```console
florian@fedora$touch ~/.ssh/config
```
 and add the following content:
```config
Host jump-host-lsv
  HostName contact.lsv.uni-saarland.de
  User {Username}
  
Host lsv-submit
  HostName submit.lsv.uni-saarland.de
  User {Username}
  ProxyJump jump-host-lsv
```
Now we can simply have access through the command
```console
florian@fedora$ssh lsv-submit
```
No password, no jumping, no host-name specification needed!
# Working with VSCode
To be fair, I am not sure anymore, whether any more steps are needed to make this work in VSCode. From what I can remember, all that you need to do now is to install the "Remote - SSH" extension. After that, an Icon to the left side bar will appear, allowing you to access the remote locations. From there, or the bottom left corner (there will be some small button/icon to click on), you can connect to the hosts.\
This will allow you to work directly on the server. All submissions will need to follow the descriptions given by Nico (https://repos.lsv.uni-saarland.de/mmosbach/htcondor-test)...