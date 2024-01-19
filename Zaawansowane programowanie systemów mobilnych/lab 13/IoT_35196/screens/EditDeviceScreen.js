// EditDeviceScreen.js

import React, { useState, useEffect } from 'react';
import { Text, View, TextInput, TouchableHighlight } from 'react-native';
import { useNavigation, useRoute } from '@react-navigation/native';
import styles from '../styles/EditDeviceStyle';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { BleManager } from 'react-native-ble-plx';
import { encode } from 'base-64';


const EditDeviceScreen = () => {
  const [name, setName] = useState('');
  const [place, setPlace] = useState('');
  const [command, setCommand] = useState('');
  const [color, setColor] = useState('');
  const Colors = ["blue", "yellow", "pink", "red", "green", "purple", "orange"];
  const navigation = useNavigation();
  const route = useRoute();
  const { deviceToEdit } = route.params || {}; 
  const bleManager = new BleManager();


  useEffect(() => {
    if (deviceToEdit) {
      setName(deviceToEdit.name || '');
      setPlace(deviceToEdit.place || '');
      setCommand(deviceToEdit.command || '');
      setColor(deviceToEdit.color || Colors[0]);
    }
  }, [deviceToEdit]);

  const handleSave = async () => {
    try {
      const updatedDevice = {
        name,
        place,
        command,
        color,
      };
      const storedDevicesList = await AsyncStorage.getItem('devicesList');
      let updatedDevicesList = storedDevicesList ? JSON.parse(storedDevicesList) : [];
  
      if (deviceToEdit) {
        updatedDevicesList = updatedDevicesList.map((device) =>
          device.id === deviceToEdit.id ? { ...device, ...updatedDevice } : device
        );
      } else {
        const newDevice = { ...updatedDevice, id: Date.now().toString() };
        updatedDevicesList.push(newDevice);
      }
      await AsyncStorage.setItem('devicesList', JSON.stringify(updatedDevicesList));
      console.log('Edycja udana');
      navigation.navigate('Devices');
    } catch (error) {
      console.error('Error saving device:', error);
    }
  };

  const exit = () => {
    navigation.navigate('Devices');
  };

  const handleColorSelect = (selectedColor) => {
    setColor(selectedColor);
  };

  const handleComend = async () => {
    try {
      const id = deviceToEdit.id;
      const serviceUUID = deviceToEdit.serviceUUID;
      const characteristicUUID = deviceToEdit.characteristicUUID;

      const connectedDevices = await bleManager.connectedDevices([serviceUUID]);
      const isConnected = connectedDevices.some(device => device.id === id);
  
      if (!isConnected) {
        const connectedDevice = await bleManager.connectToDevice(id, {autoConnect: true});
        await connectedDevice.discoverAllServicesAndCharacteristics();
      }
  
      const response = await bleManager.writeCharacteristicWithResponseForDevice(
        id, serviceUUID, characteristicUUID, encode(command)
      );
      console.log('Response ', response);
    } catch (error) {
      console.log('Error', error);
    }
  };
  

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        placeholder="Name"
        placeholderTextColor={styles.placeholder.color}
        value={name}
        onChangeText={(text) => setName(text)}
      />
      <TextInput
        style={styles.input}
        placeholder="Place"
        placeholderTextColor={styles.placeholder.color}
        value={place}
        onChangeText={(text) => setPlace(text)}
      />
      <TextInput
        style={styles.input}
        placeholder="Command"
        placeholderTextColor={styles.placeholder.color}
        value={command}
        onChangeText={(text) => setCommand(text)}
      />
      <Text style={styles.idText}>ID: {deviceToEdit.id}</Text>

      <View><Text style={{color: 'grey', fontSize:20, marginVertical:12 }} >Colors</Text></View>
      <View style={styles.colorsContainer}>
        {Colors.map((c) => (
          <TouchableHighlight
            key={c}
            style={[
              styles.circleOneColor,
              { backgroundColor: c, borderWidth: color === c ? 2 : 0 },
            ]}
            onPress={() => handleColorSelect(c)}
          >
            <View style={{ width: 40, height: 40, borderRadius: 20 }} />
          </TouchableHighlight>
        ))}
      </View>

      <TouchableHighlight
        style={[styles.buttonContainer, styles.button, { backgroundColor: 'blue' }]}
        onPress={handleComend}
      >
        <Text style={styles.buttonText}>Send Comend</Text>
      </TouchableHighlight>

      <TouchableHighlight
        style={[styles.buttonContainer, styles.button, { backgroundColor: 'green' }]}
        onPress={handleSave}
      >
        <Text style={styles.buttonText}>Save</Text>
      </TouchableHighlight>

      <TouchableHighlight
        style={[styles.buttonContainer, styles.button, { backgroundColor: 'red' }]}
        onPress={exit}
      >
        <Text style={styles.buttonText}>Cancel</Text>
      </TouchableHighlight>
    </View>
  );
};

export default EditDeviceScreen;