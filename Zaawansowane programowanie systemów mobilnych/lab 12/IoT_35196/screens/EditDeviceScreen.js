import React, { Component } from 'react';
import { Text, View, Button } from 'react-native';
import { BleManager } from 'react-native-ble-plx';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { NavigationContainer, CommonActions } from '@react-navigation/native';

import styles from '../styles/ConnectionStyle';

class ConnectionScreen extends Component {
  constructor(props) {
    super(props);
    this.manager = new BleManager();
    this.state = {
      device: null,
    };
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
        console.log('Error', error);
        return;
      }
      console.log('DEVICE ', device);

      AsyncStorage.getItem('devicesList').then((storedDevices) => {
        const devicesList = storedDevices ? JSON.parse(storedDevices) : [];
        const foundDevice = devicesList.find(dev => dev.name === device.name);
        if (foundDevice) {
          this.manager.stopDeviceScan();
          this.connectToDevice(device);
        }
      });
    });
  }

  connectToDevice(device) {
    device.connect()
      .then((connectedDevice) => {
        return connectedDevice.discoverAllServicesAndCharacteristics();
      })
      .then((characteristic) => {
        console.log('Connected to', device.name, characteristic);
        this.setState({ device: characteristic });

        AsyncStorage.getItem('devicesList').then((storedDevices) => {
          const devicesList = storedDevices ? JSON.parse(storedDevices) : [];
          const foundDevice = devicesList.find(dev => dev.name === device.name);
          const deviceInfo = {
            id: foundDevice.id,
            serviceUUID: 'FFE0', 
            characteristicUUID: 'FFE1',
          };

          AsyncStorage.setItem('device', JSON.stringify(deviceInfo));

          NavigationContainer.dispatch({
            ...CommonActions.navigate({ name: 'Devices' }),
            target: this.props.navigation.dangerouslyGetState().key,
          });
        });
      })
      .catch((error) => {
        console.log('Error', error);
      });
  }

  render() {
    return (
      <View>
        <Button
          title="Scan and Connect"
          onPress={() => this.checkBluetoothState()}
          style={styles.button}
        />
        <Text>{this.state.device ? `Connected to ${this.state.device.name}` : 'Not connected'}</Text>
      </View>
    );
  }
}

export default ConnectionScreen;
