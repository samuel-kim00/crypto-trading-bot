#!/bin/bash

# Get the current username
CURRENT_USER=$(whoami)

# Update the service file with the current username
sed -i '' "s/YOUR_USERNAME/$CURRENT_USER/g" trading_bot.service

# Copy the service file to systemd directory
sudo cp trading_bot.service /Library/LaunchDaemons/

# Reload systemd
sudo launchctl load /Library/LaunchDaemons/trading_bot.service

echo "Trading bot service has been installed and started."
echo "The bot will now run automatically when your computer starts."
echo "You can check the status with: sudo launchctl list | grep trading_bot"
echo "Logs are available in: /Users/$CURRENT_USER/Cursor Trading bot/scheduler.log" 