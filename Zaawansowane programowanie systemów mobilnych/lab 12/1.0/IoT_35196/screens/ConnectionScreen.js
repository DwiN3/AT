import React, { Component } from 'react';
import { View, Text, Button , TouchableOpacity } from 'react-native';
import { BleManager } from 'react-native-ble-plx';
import AsyncStorage from '@react-native-async-storage/async-storage';
import styles from '../styles/ConnectionStyle';

class ConnectScreen extends Component {
  constructor(props) {
    super(props);
    this.manager = new BleManager();
  }

  componentDidMount() {
    this.checkBluetoothState();
  }

  checkBluetoothState() {
    const subscription = this.manager.onStateChange((state) => {
      if (state === 'PoweredOn') {
        this.scanAndConnect();
        subscription.remove();
      }
    }, true);
  }

  scanAndConnect() {
    this.manager.startDeviceScan(null, null, (error, device) => {
      if (error) {
        console.log('Error during scanning:', error);
        return;
      }

      console.log('Found device:', device);
      if (device.name === 'MLT-BT05') {
        this.manager.stopDeviceScan();
        device.connect()
          .then((connectedDevice) => {
            return connectedDevice.discoverAllServicesAndCharacteristics();
          })
          .then((characteristic) => {
            console.log('Device connected and characteristics discovered:', characteristic);

            const deviceInfo = {
              id: device.id,
              serviceUUID: 'FFE0', 
              characteristicUUID: 'FFE1', 
            };

            AsyncStorage.setItem('device', JSON.stringify(deviceInfo)).then(() => {
              this.props.navigation.navigate('Devices');
            });
          })
          .catch((error) => {
            console.log('Error connecting to device:', error);
          });
      }
    });
  }

  changeDevice(command) {
    AsyncStorage.getItem('device').then(device => {
      device = JSON.parse(device);
      if (device) {
        this.manager.writeCharacteristicWithResponseForDevice(
          device.id, device.serviceUUID, device.characteristicUUID, btoa(command)
        ).then(response => {
          console.log('response', response);
        }).catch((error) => {
          console.log('Error', error);
        });
      }
    });
  }

  render() {
    return (
      <View>
        <Text>Connect Screen</Text>
        <TouchableOpacity
          style={styles.buttonScan}
          onPress={() => this.scanAndConnect()}
        >
          <Text style={styles.buttonText}>Scan and Connect</Text>
        </TouchableOpacity>

        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={[styles.button, styles.redButton]}
            onPress={() => this.changeDevice('red')}
          >
            <Text style={styles.buttonText}>Send Red Command</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.button, styles.greenButton]}
            onPress={() => this.changeDevice('green')}
          >
            <Text style={styles.buttonText}>Send Green Command</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.button, styles.blueButton]}
            onPress={() => this.changeDevice('blue')}
          >
            <Text style={styles.buttonText}>Send Blue Command</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.button, styles.offButton]}
            onPress={() => this.changeDevice('off')}
          >
            <Text style={styles.buttonText}>Turn Off Command</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }
}


export default ConnectScreen;