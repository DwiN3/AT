import React, { Component } from 'react';
import { View, Text, TouchableOpacity, FlatList, Image } from 'react-native';
import { BleManager } from 'react-native-ble-plx';
import AsyncStorage from '@react-native-async-storage/async-storage';
import styles from '../styles/ConnectionStyle';
import { encode } from 'base-64';

class ConnectionScreen extends Component {
  
  constructor(props) {
    super(props);
    this.manager = new BleManager();
    this.state = {
      scannedDevicesList: [],
      showCommandButtons: false,
    };
  }
  id_connect = 1;
  serviceUUID_connect = "";
  characteristicUUID_connect = "";

  checkBluetoothState() {
    const subscription = this.manager.onStateChange((state) => {
      if (state === 'PoweredOn') {
        this.scanDevices();
        subscription.remove();
      }
    }, true);
  }

  scanDevices() {
    this.showCommandButtons(false);
    this.manager.startDeviceScan(null, null, (error, device) => {
      if (error) {
        console.log('error', error);
        return;
      }
      this.handleScannedDevice(device);
    });
  }

  addDeviceMLT() {
    this.manager.startDeviceScan(null, null, async (error, device) => {
      if (error) {
        console.log('error', error);
        return;
      }
  
      //console.log('DEVICE', device);
  
      const existingDeviceIndex = this.state.scannedDevicesList.findIndex((d) => d.id === device.id);
  
      if (existingDeviceIndex === -1) {
        this.handleScannedDevice(device);
      }
  
      // My headphones: BLE EF1020 / MLT-BT05
      if (device.name === 'GR_3') {
        this.manager.stopDeviceScan();
        this.setState({ scannedDevicesList: [] });
  
        try {
          const connectedDevice = await device.connect();
          await connectedDevice.discoverAllServicesAndCharacteristics();

          const services = await connectedDevice.services();
          const serviceUUID = services[0].uuid;
          const characteristics = await connectedDevice.characteristicsForService(serviceUUID)
          const characteristicUUID = characteristics[0].uuid;

          console.log('serviceUUID: ' + serviceUUID + '\ncharacteristicUUID: ' + characteristicUUID);
          // console.log('serviceUUID: '+connectedDevice.serviceUUID+'\ncharacteristicUUID: '+connectedDevice.characteristicUUID);

          id_connect = connectedDevice.id;
          serviceUUID_connect = 'FFE0';
          characteristicUUID_connect = 'FFE1';
          // serviceUUID_connect = 'FFE0';
          // characteristicUUID_connect = 'FFE1';

          const deviceInfo = {
            id: connectedDevice.id,
            serviceUUID: serviceUUID_connect,
            characteristicUUID: characteristicUUID_connect,
          };
  
          this.showCommandButtons(true);
          this.handleSaveDevice(deviceInfo.id, device.name, serviceUUID_connect, characteristicUUID_connect);
          console.log('MLT-BT05 is Added');

      } catch (error) {
        console.log('Error', error);
      }
      }
    });
  }

  showCommandButtons(show) {
    this.setState({ showCommandButtons: show });
  }

  handleSaveDevice = async (deviceId, deviceName, serviceUUID_, characteristicUUID_) => {
    try {
      const existingDevicesString = await AsyncStorage.getItem('devicesList');
      const existingDevices = existingDevicesString ? JSON.parse(existingDevicesString) : [];
      const existingDevice = existingDevices.find((d) => d.id === deviceId);

      if (!existingDevice) {
        const newDevice = { id: deviceId, name: deviceName, color: 'yellow', serviceUUID: serviceUUID_, characteristicUUID: characteristicUUID_};
        existingDevices.push(newDevice);
        await AsyncStorage.setItem('devicesList', JSON.stringify(existingDevices));
      }
    } catch (error) {
      console.error('Error saving device:', error);
    }
  };

  handleScannedDevice(device) {
    const { id, name } = device;
    const existingDeviceIndex = this.state.scannedDevicesList.findIndex((d) => d.id === id);

    if (existingDeviceIndex === -1) {
      const newDeviceName = name || `Device ${this.state.scannedDevicesList.length + 1}`;
      const newScannedDevicesList = [...this.state.scannedDevicesList, { id, name: newDeviceName }];
      this.setState({ scannedDevicesList: newScannedDevicesList });
    }
  }

  changeDevice(command) {
        this.manager.writeCharacteristicWithResponseForDevice(
          id_connect, serviceUUID_connect, characteristicUUID_connect, encode(command)
        ).then(response => {
          console.log('response', response);
        }).catch((error) => {
          console.log('Error', error);
        });
  }


  render() {
    return (
      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={[styles.buttonScan, { marginTop: 10 }]}
          onPress={() => this.scanDevices()}
        >
          <Text style={styles.buttonText}>Scan Device</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.buttonScan}
          onPress={() => this.addDeviceMLT()}
        >
          <Text style={styles.buttonText}>Connect with MLT-BT05</Text>
        </TouchableOpacity>
        <View style={[styles.buttonContainer, styles.commandButtonsContainer]}>
          {this.state.showCommandButtons && ( 
            <>
              <TouchableOpacity
                style={[styles.buttonCommends, styles.redButton]}
                onPress={() => this.changeDevice('red')}
              >
                <Text style={styles.buttonText}>Red</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.buttonCommends, styles.greenButton]}
                onPress={() => this.changeDevice('green')}
              >
                <Text style={styles.buttonText}>Green</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.buttonCommends, styles.blueButton]}
                onPress={() => this.changeDevice('blue')}
              >
                <Text style={styles.buttonText}>Blue</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.buttonCommends, styles.offButton]}
                onPress={() => this.changeDevice('off')}
              >
                <Text style={styles.buttonText}>Turn Off</Text>
              </TouchableOpacity>
            </>
          )}
        </View>
        
        <FlatList
          data={this.state.scannedDevicesList}
          renderItem={({ item }) => (
            <TouchableOpacity onPress={() => this.handleSaveDevice(item.id, item.name)}>
              <View style={styles.itemContainer}>
                <Text style={styles.itemName}>{item.name}</Text>
                <Text style={styles.itemText}>{item.id}</Text>
              </View>
            </TouchableOpacity>
          )}
          numColumns={2}
          keyExtractor={(item) => item.id}
        />
      </View>
    );
  }
}

export default ConnectionScreen;