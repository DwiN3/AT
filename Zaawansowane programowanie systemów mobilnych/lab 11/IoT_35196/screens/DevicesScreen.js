// DeviceScreen.js

import React, { useEffect, useState, useCallback } from 'react';
import { Text, View, FlatList, TouchableOpacity, RefreshControl, Image } from 'react-native';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import AsyncStorage from '@react-native-async-storage/async-storage'; 
import styles from '../styles/DevicesStyle';

const DevicesScreen = () => {
  const navigation = useNavigation();
  const [devicesList, setDevicesList] = useState([]);
  const [refreshing, setRefreshing] = useState(false);

  const extendedDevicesList = [...devicesList, { id: '+', name: '+', place: '' }];

  useEffect(() => {
    loadDevices();
  }, []);

  useFocusEffect(
    useCallback(() => {
      const refreshData = async () => {
        setRefreshing(true);
        await loadDevices();
        setRefreshing(false);
      };

      refreshData();
    }, [])
  );

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadDevices();
    setRefreshing(false);
  }, []);

  const loadDevices = async () => {
    try {
      const storedDevices = await AsyncStorage.getItem('devicesList');
      if (storedDevices) {
        setDevicesList(JSON.parse(storedDevices));
      }
    } catch (error) {
      console.error('Error loading devices:', error);
    }
  };

  const saveDevices = async (updatedDevicesList) => {
    try {
      await AsyncStorage.setItem('devicesList', JSON.stringify(updatedDevicesList));
    } catch (error) {
      console.error('Error saving devices:', error);
    }
  };

  const clearAllData = async () => {
    try {
      await AsyncStorage.clear();
      console.log('AsyncStorage cleared successfully');
    } catch (error) {
      console.error('Error clearing AsyncStorage:', error);
    }
  };

  const deleteItem = async (index) => {
    const updatedDevicesList = [...devicesList];
    updatedDevicesList.splice(index, 1);
    setDevicesList(updatedDevicesList);
    await saveDevices(updatedDevicesList);
  };

  const renderItem = ({ item, index }) => {
    const isLastItem = index === extendedDevicesList.length - 1;
    const backgroundColor = isLastItem ? 'white' : item.color;
    const itemContainerStyle = {
      ...styles.itemContainer,
      backgroundColor: backgroundColor,
    };
    const itemNameStyle = isLastItem ? styles.lastItemName : styles.itemName;
    const itemTextStyle = isLastItem ? styles.lastItemText : styles.itemText;

    const onPressItem = () => {
      if (isLastItem) {
        navigation.navigate('New Device');
      } else {
        navigation.navigate('Edit Device', { deviceToEdit: item });
      }
    };

    const onPressToDelete = () => {
      deleteItem(index);
    };
  
    return (
      <TouchableOpacity onPress={onPressItem}>
        <View style={itemContainerStyle}>
          {!isLastItem && (
            <TouchableOpacity onPress={onPressToDelete}>
              <View style={styles.deleteItemX}>
                <Image
                  source={require('../img/icon_x.png')}
                  style={{ width: 38, height: 38 }}
                />
              </View>
            </TouchableOpacity>
          )}
          <Text style={itemNameStyle}>{item.name}</Text>
          <Text style={itemTextStyle}>{item.place}</Text>
        </View>
      </TouchableOpacity>
    );
  };

  return (
    <View style={styles.container}>
      <FlatList
        data={extendedDevicesList}
        renderItem={renderItem}
        keyExtractor={(item) => item.id}
        numColumns={2}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      />
    </View>
  );
};

export default DevicesScreen;