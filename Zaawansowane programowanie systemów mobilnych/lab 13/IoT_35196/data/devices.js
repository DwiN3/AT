class devices {
    constructor( id, name, place, command, color, serviceUUID, characteristicUUID) {
      this.id = id;
      this.name = name;
      this.place = place;
      this.command = command;
      this.color = color;
      this.serviceUUID = serviceUUID;
      this.characteristicUUID = characteristicUUID;
    }
  }

const devicesList = [

];

export { devices, devicesList };