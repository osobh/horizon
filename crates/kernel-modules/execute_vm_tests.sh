#!/usr/bin/expect -f
# Automated VM testing script

set timeout 60
set password "clouddev249"

# First, copy the archive
spawn scp -P 2222 gpu_dma_lock.tar.gz osobh@localhost:~/
expect {
    "password:" {
        send "$password\r"
        expect eof
    }
    "yes/no" {
        send "yes\r"
        expect "password:"
        send "$password\r"
        expect eof
    }
}

# Copy the test script
spawn scp -P 2222 kernel_test_all.sh osobh@localhost:~/
expect {
    "password:" {
        send "$password\r"
        expect eof
    }
}

# Now SSH and run tests
spawn ssh -p 2222 osobh@localhost
expect {
    "password:" {
        send "$password\r"
    }
}

expect "$ "
send "chmod +x kernel_test_all.sh\r"
expect "$ "
send "./kernel_test_all.sh\r"

# Interact with the session
interact